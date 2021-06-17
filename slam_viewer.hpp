#ifndef SLAM_VIEWER_HPP
#define SLAM_VIEWER_HPP

#include "opencv_viewer_data_publisher.hpp"
#include "../commandline/draw_gl.hpp"
#include "slam_implementation.hpp"
#include "viewer_data_publisher.hpp"
#include "../codegen/output/cmd_parameters.hpp"
#include "../commandline/command_queue.hpp"

#include <memory>
#include <mutex>

#include <Eigen/Dense>
#include <pangolin/pangolin.h>

namespace slam {

namespace viewer {

struct AtlasControl {
    std::unique_ptr<pangolin::Var<float>> angle;
    std::unique_ptr<pangolin::Var<float>> x;
    std::unique_ptr<pangolin::Var<float>> y;
};

class Viewer {
public:
    Viewer(const cmd::Parameters &parameters, CommandQueue &commands);

    void setup();
    void draw();

    /**
     * Request to terminate the viewer
     * (NOTE: this function does not wait for terminate)
     */
    void request_terminate();

    ViewerDataPublisher &get_data_publisher() { return dataPublisher; }

    bool is_paused() const { return menu_paused_atomic.load(); }

private:

    void create_menu_panel();

    /**
     * Update SLAM atlas if it was changed.
     */
    void getAtlas();

    /**
     * Follow to the specified camera pose
     * @param gl_cam_pose_wc
     */
    void follow_camera(const pangolin::OpenGlMatrix &gl_cam_pose_wc);

    void drawKeyframes();

    void drawLoops();

    void drawMapPoints();

    OpenCVViewerDataPublisher dataPublisher;

    /**
     * Drawing displacement for atlas maps.
     */
    Eigen::Vector2f atlasOffset(size_t mapInd) const;

    cmd::ParametersViewer parameters;
    CommandQueue &commands;

    const float viewpoint_x_, viewpoint_y_, viewpoint_z_, viewpoint_f_;

    const float keyfrm_line_width_;
    const float graph_line_width_;
    const float point_size_;
    const float camera_size_;
    draw::Theme theme;
    int theme_ind;

    pangolin::View d_cam;
    pangolin::OpenGlMatrix gl_cam_pose_wc;

    // menu panel
    std::unique_ptr<pangolin::Var<bool>> menu_paused_;
    std::unique_ptr<pangolin::Var<bool>> menu_follow_camera_;
    std::unique_ptr<pangolin::Var<bool>> menu_grid_;
    std::unique_ptr<pangolin::Var<bool>> menu_show_keyfrms_;
    std::unique_ptr<pangolin::Var<bool>> menu_show_orig_poses_;
    std::unique_ptr<pangolin::Var<bool>> menu_show_mps_;
    std::unique_ptr<pangolin::Var<bool>> menu_show_stereo_pc_;
    std::unique_ptr<pangolin::Var<bool>> menu_show_local_map_;
    std::unique_ptr<pangolin::Var<bool>> menu_show_graph_;
    std::unique_ptr<pangolin::Var<bool>> menu_show_loops_;
    std::unique_ptr<pangolin::Var<bool>> menu_show_loop_cands_;
    std::unique_ptr<pangolin::Var<bool>> menu_normal_colors_;
    std::unique_ptr<pangolin::Var<bool>> menu_natural_colors_;
    std::unique_ptr<pangolin::Var<float>> menu_mp_size_;
    std::unique_ptr<pangolin::Var<float>> menu_mp_alpha_;
    std::unique_ptr<pangolin::Var<float>> menu_map_scale_;
    std::unique_ptr<pangolin::Var<bool>> menu_change_theme_;

    std::atomic_bool menu_paused_atomic;

    // camera renderer
    std::unique_ptr<pangolin::OpenGlRenderState> s_cam_;

    // current state
    bool follow_camera_ = true;

    // viewer appearance
    const std::string map_viewer_name_{"Map Viewer"};
    static constexpr float map_viewer_width_ = 1024;
    static constexpr float map_viewer_height_ = 768;

    // Data state which is not updated on every draw call.
    std::unique_ptr<ViewerAtlas> atlas;
    std::vector<AtlasControl> atlasControls;
    double atlasOffsetX;
    double atlasOffsetY;

    draw::Animation animation;
    std::set<std::pair<MapKf, double>> animatedLoopStages;

    //-----------------------------------------
    // management for terminate process

    //! mutex for access to terminate procedure
    mutable std::mutex mtx_terminate_;

    /**
     * Check if termination is requested or not
     * @return
     */
    bool terminate_is_requested();

    //! flag which indicates termination is requested or not
    bool terminate_is_requested_ = false;
};

} // namespace viewer

} // namespace slam


#endif //SLAM_VIEWER_HPP
