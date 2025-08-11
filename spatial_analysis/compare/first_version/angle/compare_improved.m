clc;
clear;
%% 对设计模型点云进行拟合
% 读取设计目标对象点云
pipe_test = load('a1.txt');
x = pipe_test(:,1);
y = pipe_test(:,2);
z = pipe_test(:,3);
pipe_designed = [x,y,z];
% 将点云写入pcd格式
designed_pointcloud = pointCloud(pipe_designed);
% 将数据写入pcd文件中,pcread需要读取pcd文件
pcwrite(designed_pointcloud,'pipe_designed.pcd','Encoding','ascii');
pipe_designed = pcread('pipe_designed.pcd');
% 进行圆柱拟合 % 设置点到圆柱面的最大距离，目的在于降低噪点对拟合圆柱的影响
maxDistance = 1;
% 执行圆柱体拟合
cylinderModel_designed = pcfitcylinder(pipe_designed,maxDistance);

%% 提取设计模型圆柱拟合的参数
% 目标对象的前后端点
designed_point1 = cylinderModel_designed.Parameters(:,1:3);
designed_point2 = cylinderModel_designed.Parameters(:,4:6);
x1 = designed_point1(1);
y1 = designed_point1(2);
z1 = designed_point1(3);
x2= designed_point2(1);
y2= designed_point2(2);
z2 = designed_point2(3);
% 求骨架线的方向向量
v1 = [x2 - x1, y2 - y1, z2 - z1];
% 拟合骨架线
designed_X = [designed_point1(:,1) designed_point2(:,1)]';
designed_Y = [designed_point1(:,2) designed_point2(:,2)]';
designed_Z = [designed_point1(:,3) designed_point2(:,3)]';
% % 求骨架线斜率
% designed_k = tgent(designed_point1(1),designed_point1(2),designed_point2(1),designed_point2(2));
% 骨架线中心点
designed_center = cylinderModel_designed.Center;
% 求骨架线的长度
designed_length = norm(designed_point1 - designed_point2);
% 求骨架线的中点高度
designed_height = designed_center(3);
% 半径
designed_redius = cylinderModel_designed.Radius;

%% 对竣工模型点云进行拟合
% 读取竣工目标对象点云
pipe_test = load('a2.txt');
x = pipe_test(:,1);
y = pipe_test(:,2);
z = pipe_test(:,3);
pipe_built = [x,y,z];
% 将点云写入pcd格式
designed_pointcloud = pointCloud(pipe_built);
% 将数据写入pcd文件中,pcread需要读取pcd文件
pcwrite(designed_pointcloud,'pipe_built.pcd', 'Encoding', 'ascii');
pipe_built = pcread('pipe_built.pcd');
% 拟合圆柱数据集1 % 设置点到圆柱面的最大距离，目的在于降低噪点对拟合圆柱的影响
maxDistance = 1;
% 执行圆柱体拟合
cylinderModel_built = pcfitcylinder(pipe_built,maxDistance);

%% 提取竣工模型圆柱拟合的参数
% 目标对象的前后端点
built_point1 = cylinderModel_built.Parameters(:,1:3);
built_point2 = cylinderModel_built.Parameters(:,4:6);
x3 = built_point1(1);
y3 = built_point1(2);
z3 = built_point1(3);
x4 = built_point2(1);
y4 = built_point2(2);
z4 = built_point2(3);
% 求骨架线的方向向量
v2 = [x4 - x3, y4 - y3, z4 - z3];
% 拟合骨架线
built_X = [built_point1(:,1) built_point2(:,1)]';
built_Y = [built_point1(:,2) built_point2(:,2)]';
built_Z = [built_point1(:,3) built_point2(:,3)]';
% 求骨架线斜率
built_k = tgent(built_point1(1),built_point1(2),built_point2(1),built_point2(2));
% 骨架线中心点
built_center = cylinderModel_built.Center;
% 骨架线的长度
built_length = norm(built_point1 - built_point2);
% 求骨架线的中点高度
built_height = built_center(3);
% 半径
built_redius = cylinderModel_built.Radius;

%% 进行比较，需要考虑比较的顺序
% 先比较两点之间的距离，d为设计圆柱中心点画球的一个阈值
ransac_d = 10; % 为RANSAC的拟合误差
sphere_r = ransac_d + designed_redius*2;
% 两点距离
points_distance = norm(built_center - designed_center);
% 求两根直线之间的夹角
% x1 = designed_point1(1);
% y1 = designed_point1(2);
% z1 = designed_point1(3);
% x2= designed_point2(1);
% y2= designed_point2(2);
% z2 = designed_point2(3);
% x3 = built_point1(1);
% y3 = built_point1(2);
% z3 = built_point1(3);
% x4 = built_point2(1);
% y4 = built_point2(2);
% z4 = built_point2(3);
% theta = acosd(dot([x1-x2,y1-y2,z2-z1],[x3-x4,y3-y4,z4-z3])/(norm([x1-x2,y1-y2,z2-z1])*norm([x3-x4,y3-y4,z4-z3])));
% 计算点积和向量的模
% 计算点积
dotProduct = dot(v1, v2);
% 计算向量 v1 的模
normV1 = norm(v1);
% 计算向量 v2 的模
normV2 = norm(v2);
% 计算夹角
cosTheta = dotProduct / (normV1 * normV2);
% 结果是弧度
theta = acos(cosTheta);
% 转换为度
thetaDegrees = 180 - rad2deg(theta);

if points_distance <= sphere_r
    disp('no updates required')
else
    disp('updates required')
    % 进行依次对比，并对有问题的进行计数
    % 比较半径
    if abs(designed_redius - built_redius) < sphere_r
        disp('correct redius')
        result_radius = [num2str(designed_redius), 'm']
%         disp(result_radius)
    else
        disp('incorrect redius')
        designed_redius = ['designed_redius：', num2str(designed_redius), 'm']
        built_redius = ['built_redius：', num2str(built_redius), 'm']
        disp(designed_redius)
%         disp(built_redius)
    end
    % 比较高度
    if abs(designed_height - built_height) < sphere_r
        disp('correct height')
        result_height = [num2str(designed_height), 'm']
%         disp(result_height)
    else
        disp('incorrect height')
        designed_height = ['designed_height：', num2str(designed_height), 'm']
        built_height = ['built_height：', num2str(built_height), 'm']
        disp(designed_height)
        disp(built_height)
        h_distance = designed_height -built_height;
        if h_distance < 0
            str = ['设计低于竣工',num2str(abs(h_distance)),'m'];
            disp(str)
        else
            str = ['设计高于竣工',num2str(abs(h_distance)),'m'];
            disp(str)
        end
    end
    % 比较长度
    if abs(designed_length - built_length) < sphere_r
        disp('correct length')
        result_length = [num2str(designed_length), 'm']
%         disp(result_length)
    else
        disp('incorrect length')
        designed_length = [num2str(designed_length), 'm']
        built_length = [num2str(built_length), 'm']
%         disp(designed_length)
%         disp(built_length)
    end
    % 比较方向
    if thetaDegrees < 10
        disp('correct angle')
    else
        disp('incorrect angle')
        intersection_angle = [num2str(thetaDegrees), '°']
%         disp(intersection_angle)
    end
end

%% 对结果进行画图
figure
pcshow(pipe_designed,"AxesVisibility","on");
hold on
pcshow(pipe_built,"AxesVisibility","on")
hold on
set(gcf,'color','w')
set(gca,'color','w')
set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
xlabel('X(m)');
ylabel('Y(m)');
zlabel('Z(m)');
grid on
axis equal
% 对骨架线画图
figure
plot3([x1 x2], [y1 y2], [z1 z2], 'k-')
hold on
plot3([x3 x4], [y3 y4], [z3 z4], 'k-')
hold on
grid on
axis equal
% 以设计圆柱中心画球
figure
[sphere_x,sphere_y,sphere_z]  = ellipsoid(designed_center(1),designed_center(2),designed_center(3),sphere_r,sphere_r,sphere_r);
surf(sphere_x,sphere_y,sphere_z,'FaceAlpha',0.1);
designed_line = line(designed_X,designed_Y,designed_Z,'linestyle',':','color','r','LineWidth',3);
grid on
axis equal