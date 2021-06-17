#include "slam_viewer.hpp"

#include "../slam/map_point.hpp"
#include "../util/logging.hpp"

#include <opencv2/highgui.hpp>
#include <memory>
#include <utility>

namespace slam {

namespace viewer {

using RGBA = std::array<float, 4>;

Viewer::Viewer(const cmd::Parameters &parameters, CommandQueue &commands) :
    dataPublisher(parameters.slam),
    parameters(parameters.viewer),
    commands(commands),
    viewpoint_x_(0.0),
    viewpoint_y_(0.1),
    viewpoint_z_(15),
    viewpoint_f_(600.0),
    keyfrm_line_width_(1.5),
    graph_line_width_(1.5),
    point_size_(5),
    camera_size_(0.01),
    theme(draw::themes[parameters.viewer.theme]),
    theme_ind(parameters.viewer.theme),
    menu_paused_atomic(false),
    atlas(),
    atlasOffsetX(0.0),
    atlasOffsetY(0.0)
{}

void Viewer::setup() {
    pangolin::CreateWindowAndBind(map_viewer_name_, 1024, 768);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Mouse handling requires enabling GL_DEPTH_TEST.
    glEnable(GL_DEPTH_TEST);

    // setup camera renderer
    s_cam_ = std::make_unique<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(map_viewer_width_, map_viewer_height_, viewpoint_f_, viewpoint_f_,
                                   map_viewer_width_ / 2, map_viewer_height_ / 2, 0.1, 1e6),
        pangolin::ModelViewLookAt(viewpoint_x_, viewpoint_y_, viewpoint_z_, 0, 0, 0, 0.0, 1.0, 0.0));

    // create map window
    auto enforce_up = pangolin::AxisDirection::AxisZ;
    d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -map_viewer_width_ / map_viewer_height_)
        .SetHandler(new pangolin::Handler3D(*s_cam_, enforce_up));

    // Register all keyboard commands
    // Note! This doesn't conflict with cv::waitKey, only one of them can detect a single keypress, there is no duplication
    for (int key : commands.getKeys()) {
        pangolin::RegisterKeyPressCallback(key, [this, key]() {
            commands.keyboardInput(key);
        });
    }

    create_menu_panel();

    gl_cam_pose_wc.SetIdentity();
}

void Viewer::draw() {
    menu_paused_atomic.store(menu_paused_ && *menu_paused_);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    follow_camera(gl_cam_pose_wc);

    // set rendering state
    d_cam.Activate(*s_cam_);
    glClearColor(theme.bg.at(0), theme.bg.at(1), theme.bg.at(2), theme.bg.at(3));

    if (*menu_grid_) {
        glColor3fv(theme.bgEmph.data());
        draw::horizontalGrid(*menu_map_scale_);
        draw::center(0.2 * *menu_map_scale_, theme);
    }

    if (pangolin::Pushed(*menu_change_theme_)) {
        theme_ind = ++theme_ind % draw::themes.size();
        theme = draw::themes[theme_ind];
    }

    getAtlas();
    drawKeyframes();
    drawLoops();
    drawMapPoints();
    animation.handle();

    pangolin::FinishFrame();
}

void Viewer::create_menu_panel() {
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    menu_paused_ = std::make_unique<pangolin::Var<bool>>("menu.Paused", parameters.viewerPaused, true);
    menu_follow_camera_ = std::make_unique<pangolin::Var<bool>>("menu.Follow camera", parameters.followCamera, true);
    menu_grid_ = std::make_unique<pangolin::Var<bool>>("menu.Grid", parameters.showGrid, true);
    menu_show_keyfrms_ = std::make_unique<pangolin::Var<bool>>("menu.Keyframes", parameters.showKeyframes, true);
    menu_show_graph_ = std::make_unique<pangolin::Var<bool>>("menu.Graph", parameters.showGraph, true);
    menu_show_orig_poses_ = std::make_unique<pangolin::Var<bool>>("menu.Odometry poses", parameters.showOdometryPoses, true);
    menu_show_mps_ = std::make_unique<pangolin::Var<bool>>("menu.Map points", parameters.showMps, true);
    menu_show_stereo_pc_ = std::make_unique<pangolin::Var<bool>>("menu.Stereo point cloud", parameters.showStereoPc, true);
    menu_show_local_map_ = std::make_unique<pangolin::Var<bool>>("menu.Local Map", parameters.showLocalMap, true);
    menu_show_loops_ = std::make_unique<pangolin::Var<bool>>("menu.Loops", parameters.showLoops, true);
    menu_show_loop_cands_ = std::make_unique<pangolin::Var<bool>>("menu.Loop candidates", parameters.showLoopCandidates, true);
    menu_normal_colors_ = std::make_unique<pangolin::Var<bool>>("menu.Normals", parameters.normalColors, true);
    menu_natural_colors_ = std::make_unique<pangolin::Var<bool>>("menu.Natural colors", parameters.naturalColors, true);
    menu_mp_size_ = std::make_unique<pangolin::Var<float>>("menu.Map point size", parameters.mpSize, 0.1, 5, true);
    menu_mp_alpha_ = std::make_unique<pangolin::Var<float>>("menu.Map point alpha", 0.5, 0, 1, false);
    menu_map_scale_ = std::make_unique<pangolin::Var<float>>("menu.Map scale", 10.0, 2, 500, true);
    menu_change_theme_ = std::make_unique<pangolin::Var<bool>>("menu.Change theme", false, false);
}

void Viewer::getAtlas() {
    std::unique_ptr<ViewerAtlas> atlas = dataPublisher.takeAtlas();
    if (atlas) {
        bool addControls = atlasControls.empty();
        // Set drawing offsets so that atlas maps don't overlap.
        // The current map can of course become big enough to overlap.
        double dxMax = 0.0;
        double dyMax = 0.0;
        for (size_t mapInd = 0; mapInd < atlas->size(); ++mapInd) {
            double xMax = -std::numeric_limits<double>::infinity();
            double yMax = -std::numeric_limits<double>::infinity();
            double xMin = std::numeric_limits<double>::infinity();
            double yMin = std::numeric_limits<double>::infinity();
            const ViewerAtlasKeyframeVector &keyframes = (*atlas)[mapInd].keyframes;
            for (const ViewerAtlasKeyframe &keyframe : keyframes) {
                double x = keyframe.poseWC(0, 3);
                double y = keyframe.poseWC(1, 3);
                if (x > xMax) xMax = x;
                if (y > yMax) yMax = y;
                if (x < xMin) xMin = x;
                if (y < yMin) yMin = y;
            }
            if (xMax > xMin && yMax > yMin) {
                double dx = xMax - xMin;
                double dy = yMax - yMin;
                if (dx > dxMax) dxMax = dx;
                if (dy > dyMax) dyMax = dy;
            }
            // Automatic positioning is quite hard, so also add controls.
            if (addControls) {
                std::stringstream ss;
                ss << "menu." << mapInd;
                atlasControls.push_back(AtlasControl {
                    .angle = std::make_unique<pangolin::Var<float>>(ss.str() + " angle", 0, -1, 1, false),
                    .x = std::make_unique<pangolin::Var<float>>(ss.str() + " x", 0, -1, 1, false),
                    .y = std::make_unique<pangolin::Var<float>>(ss.str() + " y", 0, -1, 1, false),
                });
            }
        }
        atlasOffsetX = 1.5 * dxMax;
        atlasOffsetY = 1.5 * dyMax;

        this->atlas = std::move(atlas);
    }
}

void Viewer::follow_camera(const pangolin::OpenGlMatrix& gl_cam_pose_wc) {
    if (*menu_follow_camera_ && follow_camera_) {
        s_cam_->Follow(gl_cam_pose_wc);
    }
    else if (*menu_follow_camera_ && !follow_camera_) {
        s_cam_->SetModelViewMatrix(pangolin::ModelViewLookAt(viewpoint_x_, viewpoint_y_, viewpoint_z_, 0, 0, 0, 0.0, 1.0, 0.0));
        s_cam_->Follow(gl_cam_pose_wc);
        follow_camera_ = true;
    }
    else if (!*menu_follow_camera_ && follow_camera_) {
        follow_camera_ = false;
    }
}

void Viewer::drawKeyframes() {
    // frustum size of keyframes
    const float w = camera_size_ * *menu_map_scale_;

    ViewerDataPublisher::KeyframeVector keyframes;
    std::map<MapKf, LoopStage> loopStages;
    double age = 0.0;
    dataPublisher.getKeyframes(keyframes, loopStages, age);

    if (*menu_show_keyfrms_) {
        glLineWidth(keyfrm_line_width_);
        for (const auto &kf : keyframes) {
            MapKf mapKf { CURRENT_MAP_ID, kf.id };
            float size = w;
            if (*menu_show_loop_cands_ && loopStages.count(mapKf)) {
                switch (loopStages.at(mapKf)) {
                    case LoopStage::QUICK_TESTS:
                        size = 3 * w;
                        break;
                    case LoopStage::MAP_POINT_MATCHES:
                    case LoopStage::ACCEPTED:
                        size = 5 * w;
                        break;
                    default:
                        break;
                };
                glColor3fv(theme.orange.data());
            }
            else if (kf.current) {
                size = 2 * w;
                glColor3fv(theme.red.data());
            }
            else if (*menu_show_local_map_ && kf.localMap) {
                glColor3fv(theme.violet.data());
            }
            else {
                glColor3fv(theme.fg.data());
            }
            draw::camera(kf.poseWC, size);
        }

        glLineWidth(graph_line_width_);
        glColor3fv(theme.fg.data());
        glBegin(GL_LINES);
        for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
            glVertex3fv(keyframes[i    ].poseWC.block<3, 1>(0, 3).data());
            glVertex3fv(keyframes[i + 1].poseWC.block<3, 1>(0, 3).data());
        }
        glEnd();
    }

    if (*menu_show_keyfrms_ && *menu_show_graph_) {
        glLineWidth(graph_line_width_);
        RGBA color = theme.green;
        color[3] = 0.4;
        glColor4fv(color.data());
        glBegin(GL_LINES);
        for (size_t i = 0; i < keyframes.size(); ++i) {
            Eigen::Vector3f currCenter = keyframes[i].poseWC.block<3, 1>(0, 3);
            for (size_t neighborInd : keyframes[i].neighbors) {
                if (neighborInd + 1 == i) continue;
                Eigen::Vector3f otherCenter = keyframes[neighborInd].poseWC.block<3, 1>(0, 3);
                glVertex3fv(currCenter.data());
                glVertex3fv(otherCenter.data());
            }
        }
        glEnd();
    }

    if (*menu_show_stereo_pc_) {
        glPointSize(point_size_ * *menu_mp_size_);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < keyframes.size(); ++i) {
            const auto &kf = keyframes[i];
            if (!kf.stereoPointCloud) continue;
            Eigen::Vector4f col((i % 2) / 2.0, (i % 3) / 3.0, (i % 5) / 5.0, *menu_mp_alpha_);
            for (size_t j=0; j<kf.stereoPointCloud->size(); ++j) {
                const Eigen::Vector3f &pointCam = kf.stereoPointCloud->at(j);
                Eigen::Vector3f pointWorld = (kf.poseWC * pointCam.homogeneous()).hnormalized();
                if (*menu_natural_colors_ && kf.stereoPointCloudColor) {
                    const auto &cvCol = kf.stereoPointCloudColor->at(j);
                    col.segment<3>(0) = Eigen::Vector3f(cvCol(2), cvCol(1), cvCol(0)) / 255.0; // BGR
                }
                glColor4fv(col.data());
                glVertex3fv(pointWorld.data());
            }
        }
        glEnd();
        glColor4f(0, 0, 0, 1); // reset alpha for Color3f calls
    }

    if (*menu_show_orig_poses_) {
        glLineWidth(keyfrm_line_width_);
        glColor3fv(theme.blue.data());
        for (const auto &kf : keyframes) {
            draw::camera(kf.origPoseWC, w);
        }

        glBegin(GL_LINES);
        for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
            glVertex3fv(keyframes[i    ].origPoseWC.block<3, 1>(0, 3).data());
            glVertex3fv(keyframes[i + 1].origPoseWC.block<3, 1>(0, 3).data());
        }
        glEnd();
    }

    if (*menu_show_keyfrms_ && atlas) {
        glLineWidth(keyfrm_line_width_);

        Eigen::Matrix4f offset = Eigen::Matrix4f::Zero();
        for (size_t mapInd = 0; mapInd < atlas->size(); ++mapInd) {
            const ViewerAtlasKeyframeVector &keyframes = (*atlas)[mapInd].keyframes;
            offset.block<2, 1>(0, 3) = atlasOffset(mapInd);
            Eigen::Matrix4f R = Eigen::Matrix4f::Identity();
            R.topLeftCorner<3, 3>() = Eigen::AngleAxisf(*atlasControls[mapInd].angle * M_PI, Eigen::Vector3f::UnitZ()).matrix();
            for (const ViewerAtlasKeyframe &keyframe : keyframes) {
                Eigen::Matrix4f pose = R * keyframe.poseWC;
                Eigen::Vector3f position = (pose + offset).block<3, 1>(0, 3);
                MapKf mapKf { MapId(mapInd), keyframe.id };
                if (loopStages.count(mapKf)) {
                    if (loopStages.at(mapKf) == LoopStage::RELOCATION_MAP_POINT_RANSAC) {
                        if (!animatedLoopStages.count({ mapKf, age })) {
                            animation.fadeAway(position, 0.07 * *menu_map_scale_, theme.orange);
                            animatedLoopStages.insert({ mapKf, age });
                        }
                    }
                }
                glColor3fv(theme.fg.data());
                draw::camera(pose + offset, w);
            }

            glColor3fv(theme.fg.data());
            glBegin(GL_LINES);
            for (size_t i = 0; i + 1 < keyframes.size(); ++i) {
                glVertex3fv((R * keyframes[i    ].poseWC + offset).block<3, 1>(0, 3).eval().data());
                glVertex3fv((R * keyframes[i + 1].poseWC + offset).block<3, 1>(0, 3).eval().data());
            }
            glEnd();
        }
    }
}

static RGBA getMapPointColor(const ViewerMapPoint &mp, const draw::Theme &theme) {
    const auto &rgba = mp.status == int(MapPointStatus::TRIANGULATED) ?
        (mp.nowVisible ? theme.red : theme.fg) :
        (mp.nowVisible ? theme.magenta : theme.violet); // violet used to have alpha 0.3
    float alphaMult = !mp.localMap ? 0.3 : 1.0;
    return {{ rgba[0], rgba[1], rgba[2], alphaMult*rgba[3] }};
}

void Viewer::drawMapPoints() {
    if (!*menu_show_mps_) {
        return;
    }

    ViewerDataPublisher::MapPointVector mapPoints = dataPublisher.getMapPoints();

    glPointSize(point_size_ * *menu_mp_size_);

    glBegin(GL_POINTS);

    // render high alpha first -> nicer result with alpha + depth buffer
    std::stable_sort(mapPoints.begin(), mapPoints.end(), [&](const ViewerMapPoint &a, const ViewerMapPoint &b) -> bool {
        return getMapPointColor(a, theme)[3] > getMapPointColor(b, theme)[3];
    });

    for (const auto &mp : mapPoints) {
        auto rgba = getMapPointColor(mp, theme);
        if (*menu_normal_colors_) {
            Eigen::Vector3f col = (mp.normal + Eigen::Vector3f(1, 1, 1)) * 0.5f;
            rgba[0] = col.x();
            rgba[1] = col.y();
            rgba[2] = col.z();
        }
        else if (*menu_natural_colors_) {
            rgba[0] = mp.color.x();
            rgba[1] = mp.color.y();
            rgba[2] = mp.color.z();
            rgba[3] = *menu_mp_alpha_;
        }
        glColor4fv(rgba.data());
        glVertex3fv(mp.position.data());
    }

    glColor4f(0, 0, 0, 1); // reset alpha for Color3f calls

    /*
    if (atlas) {
        glColor3fv(theme.fg.data());
        Eigen::Vector3f offset = Eigen::Vector3f::Zero();
        for (size_t mapInd = 0; mapInd < atlas->size(); ++mapInd) {
            const ViewerAtlasMapPointVector &mapPoints = (*atlas)[mapInd].mapPoints;
            offset.segment<2>(0) = atlasOffset(mapInd);
            for (const ViewerAtlasMapPoint &mapPoint : mapPoints) {
                glVertex3dv((mapPoint.position + offset).eval().data());
            }
        }
    }
    */

    glEnd();
}

void Viewer::drawLoops() {
    if (!*menu_show_loops_) {
        return;
    }

    const float w = camera_size_ * *menu_map_scale_;

    glLineWidth(3.0);
    std::vector<ViewerLoopClosure> lcs = dataPublisher.getLoopClosures();
    for (size_t i = 0; i < lcs.size(); i++) {
        auto loopClosure = lcs.at(i);
        Eigen::Vector3f current = loopClosure.currentPose.block<3, 1>(0, 3);
        Eigen::Vector3f candidate = loopClosure.candidatePose.block<3, 1>(0, 3);
        Eigen::Vector3f updated = loopClosure.updatedPose.block<3, 1>(0, 3);
        Eigen::Matrix4f updatedPose = loopClosure.updatedPose;

        glColor3fv(theme.orange.data());
        glBegin(GL_LINES);
        draw::line(candidate, updated);
        draw::line(current, updated);
        glEnd();

        Eigen::Affine3f affine3F(updatedPose);
        draw::camera(affine3F.matrix(), w);
    }
}

void Viewer::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}

bool Viewer::terminate_is_requested() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

Eigen::Vector2f Viewer::atlasOffset(size_t mapInd) const {
    // Lay maps in two rows.
    size_t n = mapInd + 1;
    double s = 2 * *menu_map_scale_;
    return {
        (n % 2) * atlasOffsetX + s * *atlasControls[mapInd].x,
        (n / 2) * atlasOffsetY + s * *atlasControls[mapInd].y
    };
}

} // namespace viewer

} // namespace slam
