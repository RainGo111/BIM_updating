% 自定义参数
% self-defined parameters
config = struct(...
    'max_distance', 1, ...           % 圆柱拟合最大距离 (m)
    'ransac_d', 10, ...             % RANSAC拟合误差 (m)
    'angle_threshold', 10, ...       % 角度差异阈值 (度)
    'radius_multiplier', 2, ...      % 半径倍数用于球体阈值计算
    'show_plots', true, ...         % 是否显示图形
    'verbose', true ...             % 是否显示详细输出
);
% results = compare_pipe_improved_en('a1.txt', 'a2.txt', config);

results = compare_pipe_improved('a1.txt', 'a2.txt', config);