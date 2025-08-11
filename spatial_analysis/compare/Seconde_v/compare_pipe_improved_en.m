function pipe_comparison_results = compare_pipe_improved_en(design_file, built_file, config)
% Input parameters:
%   design_file: Design pipe point cloud file path
%   built_file: As-built pipe point cloud file path  
%   config: Configuration structure containing various threshold parameters

clc;

%% Parameter Configuration (use default values if config not provided)
if nargin < 3
    config = struct();
end

% Set default configuration parameters
default_config = struct(...
    'max_distance', 1, ...           % Maximum distance for cylinder fitting
    'ransac_d', 10, ...             % RANSAC fitting error threshold
    'angle_threshold', 10, ...       % Angle difference threshold (degrees)
    'radius_multiplier', 2, ...      % Radius multiplier for sphere threshold calculation
    'show_plots', true, ...         % Whether to display plots
    'verbose', true ...             % Whether to show detailed output
);

config = merge_configs(default_config, config);

%% Data Reading and Preprocessing
try
    fprintf('Reading design pipe data...\n');
    designed_params = process_pipe_data(design_file, 'designed', config);
    
    fprintf('Reading as-built pipe data...\n');
    built_params = process_pipe_data(built_file, 'built', config);
    
catch ME
    error('Data reading failed: %s', ME.message);
end

%% Perform Comparison Analysis
comparison_results = perform_comparison(designed_params, built_params, config);

%% Generate Output Results
pipe_comparison_results = generate_results(designed_params, built_params, comparison_results, config);

%% Visualize Results
if config.show_plots
    visualize_results(designed_params, built_params, config);
end

%% Print Results Report
if config.verbose
    print_comparison_report(pipe_comparison_results);
end

end

%% Helper Functions

function merged_config = merge_configs(default_config, user_config)
% Merge default configuration with user configuration
merged_config = default_config;
if ~isempty(user_config)
    fields = fieldnames(user_config);
    for i = 1:length(fields)
        merged_config.(fields{i}) = user_config.(fields{i});
    end
end
end

function pipe_params = process_pipe_data(filename, pipe_type, config)
% Process pipe point cloud data and extract parameters
try
    % Read point cloud data
    pipe_data = load(filename);
    if size(pipe_data, 2) < 3
        error('Point cloud data format error, at least 3 columns (x,y,z) required');
    end
    
    points = pipe_data(:, 1:3);
    
    % Create point cloud object
    point_cloud = pointCloud(points);
    
    % Generate temporary pcd file
    temp_filename = sprintf('temp_%s_pipe.pcd', pipe_type);
    pcwrite(point_cloud, temp_filename, 'Encoding', 'ascii');
    pipe_cloud = pcread(temp_filename);
    
    % Clean up temporary file
    if exist(temp_filename, 'file')
        delete(temp_filename);
    end
    
    % Perform cylinder fitting
    cylinder_model = pcfitcylinder(pipe_cloud, config.max_distance);
    
    % Extract parameters
    pipe_params = extract_cylinder_parameters(cylinder_model);
    pipe_params.type = pipe_type;
    pipe_params.point_cloud = pipe_cloud;
    
catch ME
    error('Error processing %s pipe data: %s', pipe_type, ME.message);
end
end

function params = extract_cylinder_parameters(cylinder_model)
% Extract geometric parameters from cylinder model
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

% Normalize direction vector
params.unit_direction = params.direction_vector / norm(params.direction_vector);
end

function results = perform_comparison(designed, built, config)
% Perform detailed comparison analysis
results = struct();

% Calculate center point distance
results.center_distance = norm(built.center - designed.center);

% Calculate sphere threshold radius
sphere_radius = config.ransac_d + designed.radius * config.radius_multiplier;
results.sphere_radius = sphere_radius;

% Determine if update is needed
results.needs_update = results.center_distance > sphere_radius;

% If update needed, perform detailed comparison
if results.needs_update
    % Radius comparison
    results.radius_diff = abs(designed.radius - built.radius);
    results.radius_correct = results.radius_diff < sphere_radius;
    
    % Height comparison
    results.height_diff = designed.height - built.height;
    results.height_correct = abs(results.height_diff) < sphere_radius;
    
    % Length comparison
    results.length_diff = abs(designed.length - built.length);
    results.length_correct = results.length_diff < sphere_radius;
    
    % Angle comparison (corrected angle calculation)
    dot_product = dot(designed.unit_direction, built.unit_direction);
    % Ensure dot product is within valid range
    dot_product = max(-1, min(1, dot_product));
    angle_rad = acos(abs(dot_product)); % Use absolute value to get acute angle
    results.angle_degrees = rad2deg(angle_rad);
    results.angle_correct = results.angle_degrees < config.angle_threshold;
else
    % If no update needed, set all parameters as correct
    results.radius_correct = true;
    results.height_correct = true;  
    results.length_correct = true;
    results.angle_correct = true;
end
end

function pipe_results = generate_results(designed, built, comparison, config)
% Generate structured comparison results
pipe_results = struct();

% Basic information
pipe_results.needs_update = comparison.needs_update;
pipe_results.center_distance = comparison.center_distance;
pipe_results.sphere_threshold = comparison.sphere_radius;

% Detailed comparison results
pipe_results.comparison = struct();

if comparison.needs_update
    % Radius comparison
    pipe_results.comparison.radius = struct(...
        'designed', designed.radius, ...
        'built', built.radius, ...
        'difference', comparison.radius_diff, ...
        'correct', comparison.radius_correct ...
    );
    
    % Height comparison
    pipe_results.comparison.height = struct(...
        'designed', designed.height, ...
        'built', built.height, ...
        'difference', comparison.height_diff, ...
        'correct', comparison.height_correct ...
    );
    
    % Length comparison
    pipe_results.comparison.length = struct(...
        'designed', designed.length, ...
        'built', built.length, ...
        'difference', comparison.length_diff, ...
        'correct', comparison.length_correct ...
    );
    
    % Angle comparison
    pipe_results.comparison.angle = struct(...
        'degrees', comparison.angle_degrees, ...
        'correct', comparison.angle_correct ...
    );
end

% Summary statistics
if comparison.needs_update
    total_checks = 4;
    correct_checks = sum([comparison.radius_correct, comparison.height_correct, ...
                         comparison.length_correct, comparison.angle_correct]);
    pipe_results.accuracy_rate = correct_checks / total_checks;
else
    pipe_results.accuracy_rate = 1.0; % 100% accuracy
end
end

function visualize_results(designed, built, config)
% Visualize comparison results
try
    % Figure 1: Point cloud comparison - using compatible pcshow syntax
    figure('Name', 'Pipe Point Cloud Comparison', 'Color', 'w');
    
    % Add red color for design point cloud
    designed_points = designed.point_cloud.Location;
    designed_colors = repmat([1 0 0], size(designed_points, 1), 1); % Red
    designed_colored = pointCloud(designed_points, 'Color', designed_colors);
    pcshow(designed_colored, 'MarkerSize', 20);
    
    hold on;
    
    % Add blue color for as-built point cloud
    built_points = built.point_cloud.Location;
    built_colors = repmat([0 0 1], size(built_points, 1), 1); % Blue
    built_colored = pointCloud(built_points, 'Color', built_colors);
    pcshow(built_colored, 'MarkerSize', 20);
    
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('Red: Design Pipe, Blue: As-built Pipe');
    grid on; axis equal;
    
    % Set axis style
    set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15]);
    
catch ME
    % If colored point cloud fails, use default display
    warning('Colored point cloud display failed, using default display: %s', ME.message);
    figure('Name', 'Pipe Point Cloud Comparison', 'Color', 'w');
    pcshow(designed.point_cloud, 'MarkerSize', 20);
    hold on;
    pcshow(built.point_cloud, 'MarkerSize', 20);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('First: Design Pipe, Second: As-built Pipe');
    grid on; axis equal;
end

% Figure 2: Skeleton line comparison
figure('Name', 'Pipe Centerline Comparison', 'Color', 'w');
plot3([designed.point1(1), designed.point2(1)], ...
      [designed.point1(2), designed.point2(2)], ...
      [designed.point1(3), designed.point2(3)], ...
      'r-', 'LineWidth', 3, 'DisplayName', 'Design Centerline');
hold on;
plot3([built.point1(1), built.point2(1)], ...
      [built.point1(2), built.point2(2)], ...
      [built.point1(3), built.point2(3)], ...
      'b-', 'LineWidth', 3, 'DisplayName', 'As-built Centerline');

% Draw center points
plot3(designed.center(1), designed.center(2), designed.center(3), ...
      'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Design Center');
plot3(built.center(1), built.center(2), built.center(3), ...
      'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b', 'DisplayName', 'As-built Center');

% Connect the two center points
plot3([designed.center(1), built.center(1)], ...
      [designed.center(2), built.center(2)], ...
      [designed.center(3), built.center(3)], ...
      'k--', 'LineWidth', 1, 'DisplayName', 'Center Distance');

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Pipe Centerlines and Center Points Comparison');
legend('show');
grid on; axis equal;

% Figure 3: Threshold sphere visualization
figure('Name', 'Design Center Threshold Sphere', 'Color', 'w');
sphere_r = config.ransac_d + designed.radius * config.radius_multiplier;
[sphere_x, sphere_y, sphere_z] = ellipsoid(designed.center(1), designed.center(2), ...
    designed.center(3), sphere_r, sphere_r, sphere_r);
surf(sphere_x, sphere_y, sphere_z, 'FaceAlpha', 0.1, 'EdgeColor', 'none', ...
     'FaceColor', 'cyan', 'DisplayName', 'Threshold Sphere');
hold on;

% Draw design centerline
plot3([designed.point1(1), designed.point2(1)], ...
      [designed.point1(2), designed.point2(2)], ...
      [designed.point1(3), designed.point2(3)], ...
      'r:', 'LineWidth', 3, 'DisplayName', 'Design Centerline');

% Draw as-built centerline
plot3([built.point1(1), built.point2(1)], ...
      [built.point1(2), built.point2(2)], ...
      [built.point1(3), built.point2(3)], ...
      'b:', 'LineWidth', 3, 'DisplayName', 'As-built Centerline');

% Annotate center distance
center_distance = norm(built.center - designed.center);
text(designed.center(1), designed.center(2), designed.center(3) + sphere_r/4, ...
     sprintf('Center Distance: %.2f m', center_distance), ...
     'HorizontalAlignment', 'center', 'FontSize', 12, 'BackgroundColor', 'white');

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('Design Pipe Center Threshold Sphere (Radius: %.2f m)', sphere_r));
legend('show');
grid on; axis equal;
end

function print_comparison_report(results)
% Print detailed comparison report
fprintf('\n=== Pipe Comparison Analysis Report ===\n');
fprintf('Center Distance: %.3f m\n', results.center_distance);
fprintf('Threshold Radius: %.3f m\n', results.sphere_threshold);

if results.needs_update
    fprintf('\nConclusion: Update Required ❌\n\n');
    
    fprintf('Detailed Comparison Results:\n');
    if isfield(results.comparison, 'radius')
        fprintf('• Radius: %s (Design: %.3f m, As-built: %.3f m, Difference: %.3f m)\n', ...
            get_status_text(results.comparison.radius.correct), ...
            results.comparison.radius.designed, ...
            results.comparison.radius.built, ...
            results.comparison.radius.difference);
    end
    
    if isfield(results.comparison, 'height')
        fprintf('• Height: %s (Design: %.3f m, As-built: %.3f m, Difference: %.3f m)\n', ...
            get_status_text(results.comparison.height.correct), ...
            results.comparison.height.designed, ...
            results.comparison.height.built, ...
            results.comparison.height.difference);
        
        if ~results.comparison.height.correct
            if results.comparison.height.difference > 0
                fprintf('  → Design is %.3f m higher than as-built\n', abs(results.comparison.height.difference));
            else
                fprintf('  → Design is %.3f m lower than as-built\n', abs(results.comparison.height.difference));
            end
        end
    end
    
    if isfield(results.comparison, 'length')
        fprintf('• Length: %s (Design: %.3f m, As-built: %.3f m, Difference: %.3f m)\n', ...
            get_status_text(results.comparison.length.correct), ...
            results.comparison.length.designed, ...
            results.comparison.length.built, ...
            results.comparison.length.difference);
    end
    
    if isfield(results.comparison, 'angle')
        fprintf('• Angle: %s (Intersection Angle: %.2f°)\n', ...
            get_status_text(results.comparison.angle.correct), ...
            results.comparison.angle.degrees);
    end
    
else
    fprintf('\nConclusion: No Update Required ✅\n');
end

fprintf('\nOverall Accuracy Rate: %.1f%%\n', results.accuracy_rate * 100);
fprintf('=====================================\n\n');
end

function status_text = get_status_text(is_correct)
% Return status text based on boolean value
if is_correct
    status_text = 'Correct ✅';
else
    status_text = 'Incorrect ❌';
end
end