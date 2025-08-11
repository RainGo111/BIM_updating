function pipe_comparison_results = compare_pipe_improved(design_file, built_file, config)
% 输入参数:
%   design_file: 设计管道点云文件路径
%   built_file: 竣工管道点云文件路径  
%   config: 配置结构体，包含各种阈值参数

clc;

%% 参数配置 (如果未提供config则使用默认值)
if nargin < 3
    config = struct();
end

% 设置默认配置参数
default_config = struct(...
    'max_distance', 1, ...           % 圆柱拟合最大距离
    'ransac_d', 10, ...             % RANSAC拟合误差
    'angle_threshold', 10, ...       % 角度差异阈值(度)
    'radius_multiplier', 2, ...      % 半径倍数用于球体阈值计算
    'show_plots', true, ...         % 是否显示图形
    'verbose', true ...             % 是否显示详细输出
);

config = merge_configs(default_config, config);

%% 数据读取和预处理
try
    fprintf('正在读取设计管道数据...\n');
    designed_params = process_pipe_data(design_file, 'designed', config);
    
    fprintf('正在读取竣工管道数据...\n');
    built_params = process_pipe_data(built_file, 'built', config);
    
catch ME
    error('数据读取失败: %s', ME.message);
end

%% 执行对比分析
comparison_results = perform_comparison(designed_params, built_params, config);

%% 生成输出结果
pipe_comparison_results = generate_results(designed_params, built_params, comparison_results, config);

%% 可视化结果
if config.show_plots
    visualize_results(designed_params, built_params, config);
end

%% 打印结果报告
if config.verbose
    print_comparison_report(pipe_comparison_results);
end

end

%% 辅助函数

function merged_config = merge_configs(default_config, user_config)
% 合并默认配置和用户配置
merged_config = default_config;
if ~isempty(user_config)
    fields = fieldnames(user_config);
    for i = 1:length(fields)
        merged_config.(fields{i}) = user_config.(fields{i});
    end
end
end

function pipe_params = process_pipe_data(filename, pipe_type, config)
% 处理管道点云数据并提取参数
try
    % 读取点云数据
    pipe_data = load(filename);
    if size(pipe_data, 2) < 3
        error('点云数据格式错误，需要至少3列(x,y,z)坐标');
    end
    
    points = pipe_data(:, 1:3);
    
    % 创建点云对象
    point_cloud = pointCloud(points);
    
    % 生成临时pcd文件
    temp_filename = sprintf('temp_%s_pipe.pcd', pipe_type);
    pcwrite(point_cloud, temp_filename, 'Encoding', 'ascii');
    pipe_cloud = pcread(temp_filename);
    
    % 清理临时文件
    if exist(temp_filename, 'file')
        delete(temp_filename);
    end
    
    % 执行圆柱拟合
    cylinder_model = pcfitcylinder(pipe_cloud, config.max_distance);
    
    % 提取参数
    pipe_params = extract_cylinder_parameters(cylinder_model);
    pipe_params.type = pipe_type;
    pipe_params.point_cloud = pipe_cloud;
    
catch ME
    error('处理%s管道数据时出错: %s', pipe_type, ME.message);
end
end

function params = extract_cylinder_parameters(cylinder_model)
% 从圆柱模型中提取几何参数
point1 = cylinder_model.Parameters(1:3);
point2 = cylinder_model.Parameters(4:6);

params = struct();
params.point1 = point1';
params.point2 = point2';
params.direction_vector = point2' - point1';
params.center = cylinder_model.Center;
params.length = norm(point2 - point1);
params.height = cylinder_model.Center(3);
params.radius = cylinder_model.Radius;

% 归一化方向向量
params.unit_direction = params.direction_vector / norm(params.direction_vector);
end

function results = perform_comparison(designed, built, config)
% 执行详细对比分析
results = struct();

% 计算中心点距离
results.center_distance = norm(built.center - designed.center);

% 计算球体阈值半径
sphere_radius = config.ransac_d + designed.radius * config.radius_multiplier;
results.sphere_radius = sphere_radius;

% 判断是否需要更新
results.needs_update = results.center_distance > sphere_radius;

% 如果需要更新，进行详细对比
if results.needs_update
    % 半径对比
    results.radius_diff = abs(designed.radius - built.radius);
    results.radius_correct = results.radius_diff < sphere_radius;
    
    % 高度对比
    results.height_diff = designed.height - built.height;
    results.height_correct = abs(results.height_diff) < sphere_radius;
    
    % 长度对比
    results.length_diff = abs(designed.length - built.length);
    results.length_correct = results.length_diff < sphere_radius;
    
    % 角度对比 (修正角度计算)
    dot_product = dot(designed.unit_direction, built.unit_direction);
    % 确保点积在有效范围内
    dot_product = max(-1, min(1, dot_product));
    angle_rad = acos(abs(dot_product)); % 使用绝对值确保得到锐角
    results.angle_degrees = rad2deg(angle_rad);
    results.angle_correct = results.angle_degrees < config.angle_threshold;
else
    % 如果不需要更新，设置所有参数为正确
    results.radius_correct = true;
    results.height_correct = true;  
    results.length_correct = true;
    results.angle_correct = true;
end
end

function pipe_results = generate_results(designed, built, comparison, config)
% 生成结构化的对比结果
pipe_results = struct();

% 基本信息
pipe_results.needs_update = comparison.needs_update;
pipe_results.center_distance = comparison.center_distance;
pipe_results.sphere_threshold = comparison.sphere_radius;

% 详细对比结果
pipe_results.comparison = struct();

if comparison.needs_update
    % 半径对比
    pipe_results.comparison.radius = struct(...
        'designed', designed.radius, ...
        'built', built.radius, ...
        'difference', comparison.radius_diff, ...
        'correct', comparison.radius_correct ...
    );
    
    % 高度对比
    pipe_results.comparison.height = struct(...
        'designed', designed.height, ...
        'built', built.height, ...
        'difference', comparison.height_diff, ...
        'correct', comparison.height_correct ...
    );
    
    % 长度对比
    pipe_results.comparison.length = struct(...
        'designed', designed.length, ...
        'built', built.length, ...
        'difference', comparison.length_diff, ...
        'correct', comparison.length_correct ...
    );
    
    % 角度对比
    pipe_results.comparison.angle = struct(...
        'degrees', comparison.angle_degrees, ...
        'correct', comparison.angle_correct ...
    );
end

% 汇总统计
if comparison.needs_update
    total_checks = 4;
    correct_checks = sum([comparison.radius_correct, comparison.height_correct, ...
                         comparison.length_correct, comparison.angle_correct]);
    pipe_results.accuracy_rate = correct_checks / total_checks;
else
    pipe_results.accuracy_rate = 1.0; % 100%准确
end
end

function visualize_results(designed, built, config)
% 可视化对比结果
try
    % 图1: 点云对比 - 使用兼容的pcshow语法
    figure('Name', '管道点云对比', 'Color', 'w');
    
    % 为设计点云添加红色
    designed_points = designed.point_cloud.Location;
    designed_colors = repmat([1 0 0], size(designed_points, 1), 1); % 红色
    designed_colored = pointCloud(designed_points, 'Color', designed_colors);
    pcshow(designed_colored, 'MarkerSize', 20);
    
    hold on;
    
    % 为竣工点云添加蓝色
    built_points = built.point_cloud.Location;
    built_colors = repmat([0 0 1], size(built_points, 1), 1); % 蓝色
    built_colored = pointCloud(built_points, 'Color', built_colors);
    pcshow(built_colored, 'MarkerSize', 20);
    
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('红色: 设计管道, 蓝色: 竣工管道');
    grid on; axis equal;
    
    % 设置坐标轴样式
    set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15]);
    
catch ME
    % 如果彩色点云失败，使用默认显示
    warning('彩色点云显示失败，使用默认显示: %s', ME.message);
    figure('Name', '管道点云对比', 'Color', 'w');
    pcshow(designed.point_cloud, 'MarkerSize', 20);
    hold on;
    pcshow(built.point_cloud, 'MarkerSize', 20);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('第一个: 设计管道, 第二个: 竣工管道');
    grid on; axis equal;
end

% 图2: 骨架线对比
figure('Name', '管道骨架线对比', 'Color', 'w');
plot3([designed.point1(1), designed.point2(1)], ...
      [designed.point1(2), designed.point2(2)], ...
      [designed.point1(3), designed.point2(3)], ...
      'r-', 'LineWidth', 3, 'DisplayName', '设计骨架线');
hold on;
plot3([built.point1(1), built.point2(1)], ...
      [built.point1(2), built.point2(2)], ...
      [built.point1(3), built.point2(3)], ...
      'b-', 'LineWidth', 3, 'DisplayName', '竣工骨架线');

% 绘制中心点
plot3(designed.center(1), designed.center(2), designed.center(3), ...
      'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', '设计中心');
plot3(built.center(1), built.center(2), built.center(3), ...
      'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'DisplayName', '竣工中心');

% 连接两个中心点
plot3([designed.center(1), built.center(1)], ...
      [designed.center(2), built.center(2)], ...
      [designed.center(3), built.center(3)], ...
      'k--', 'LineWidth', 1, 'DisplayName', '中心距离');

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('管道骨架线和中心点对比');
legend('show');
grid on; axis equal;

% 图3: 阈值球体可视化
figure('Name', '设计中心阈值球体', 'Color', 'w');
sphere_r = config.ransac_d + designed.radius * config.radius_multiplier;
[sphere_x, sphere_y, sphere_z] = ellipsoid(designed.center(1), designed.center(2), ...
    designed.center(3), sphere_r, sphere_r, sphere_r);
surf(sphere_x, sphere_y, sphere_z, 'FaceAlpha', 0.1, 'EdgeColor', 'none', ...
     'FaceColor', 'cyan', 'DisplayName', '阈值球体');
hold on;

% 绘制设计骨架线
plot3([designed.point1(1), designed.point2(1)], ...
      [designed.point1(2), designed.point2(2)], ...
      [designed.point1(3), designed.point2(3)], ...
      'r:', 'LineWidth', 3, 'DisplayName', '设计骨架线');

% 绘制竣工骨架线
plot3([built.point1(1), built.point2(1)], ...
      [built.point1(2), built.point2(2)], ...
      [built.point1(3), built.point2(3)], ...
      'b:', 'LineWidth', 3, 'DisplayName', '竣工骨架线');

% 标注中心点距离
center_distance = norm(built.center - designed.center);
text(designed.center(1), designed.center(2), designed.center(3) + sphere_r/4, ...
     sprintf('中心距离: %.2f m', center_distance), ...
     'HorizontalAlignment', 'center', 'FontSize', 12, 'BackgroundColor', 'white');

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('设计管道中心阈值球体 (半径: %.2f m)', sphere_r));
legend('show');
grid on; axis equal;
end

function print_comparison_report(results)
% 打印详细的对比报告
fprintf('\n=== 管道对比分析报告 ===\n');
fprintf('中心点距离: %.3f m\n', results.center_distance);
fprintf('阈值半径: %.3f m\n', results.sphere_threshold);

if results.needs_update
    fprintf('\n结论: 需要更新 ❌\n\n');
    
    fprintf('详细对比结果:\n');
    if isfield(results.comparison, 'radius')
        fprintf('• 半径: %s (设计: %.3f m, 竣工: %.3f m, 差异: %.3f m)\n', ...
            get_status_text(results.comparison.radius.correct), ...
            results.comparison.radius.designed, ...
            results.comparison.radius.built, ...
            results.comparison.radius.difference);
    end
    
    if isfield(results.comparison, 'height')
        fprintf('• 高度: %s (设计: %.3f m, 竣工: %.3f m, 差异: %.3f m)\n', ...
            get_status_text(results.comparison.height.correct), ...
            results.comparison.height.designed, ...
            results.comparison.height.built, ...
            results.comparison.height.difference);
        
        if ~results.comparison.height.correct
            if results.comparison.height.difference > 0
                fprintf('  → 设计高于竣工 %.3f m\n', abs(results.comparison.height.difference));
            else
                fprintf('  → 设计低于竣工 %.3f m\n', abs(results.comparison.height.difference));
            end
        end
    end
    
    if isfield(results.comparison, 'length')
        fprintf('• 长度: %s (设计: %.3f m, 竣工: %.3f m, 差异: %.3f m)\n', ...
            get_status_text(results.comparison.length.correct), ...
            results.comparison.length.designed, ...
            results.comparison.length.built, ...
            results.comparison.length.difference);
    end
    
    if isfield(results.comparison, 'angle')
        fprintf('• 角度: %s (夹角: %.2f°)\n', ...
            get_status_text(results.comparison.angle.correct), ...
            results.comparison.angle.degrees);
    end
    
else
    fprintf('\n结论: 无需更新 ✅\n');
end

fprintf('\n总体准确率: %.1f%%\n', results.accuracy_rate * 100);
fprintf('======================\n\n');
end

function status_text = get_status_text(is_correct)
% 根据布尔值返回状态文本
if is_correct
    status_text = '正确 ✅';
else
    status_text = '不正确 ❌';
end
end