clc;
clear;
%% 对设计模型点云进行拟合
% 读取数据集1
pipe_test = load('a1.txt');
x = pipe_test(:,1);
y = pipe_test(:,2);
z = pipe_test(:,3);
pipe_designed = [x,y,z];
% % 点云可视化
% figure(1);
% pcshow([x,y,z]);
% title('pipe_designed');
% xlabel('X(m)');
% zlabel('Z(m)');
% ylabel('Y(m)');
% 将点云写入pcd格式
designed_pointcloud = pointCloud(pipe_designed);
% 将数据写入pcd文件中,pcread需要读取pcd文件
pcwrite(designed_pointcloud,'pipe_designed.pcd','Encoding','ascii');
pipe_designed = pcread('pipe_designed.pcd');
% 进行圆柱拟合
% 设置点到圆柱面的最大距离，目的在于降低噪点对拟合圆柱的影响
maxDistance = 1;
% 执行圆柱体拟合
cylinderModel_designed = pcfitcylinder(pipe_designed,maxDistance);
% % 画出拟合的结果
% figure(2)
% pcshow(piep_designed.Location)
% title('圆柱1拟合效果')
% xlabel('X(m)');
% ylabel('Y(m)');
% zlabel('Z(m)');
% hold on
% plot(cylinderModel_designed)

%% 提取设计模型圆柱拟合的参数
% 三维性质：半径
designed_redius = cylinderModel_designed.Radius;
% 三维性质：圆柱中心点
designed_center = cylinderModel_designed.Center;
% 二维性质：中心轴线的长度
designed_point1 = cylinderModel_designed.Parameters(:,1:3);
designed_point2 = cylinderModel_designed.Parameters(:,4:6);
designed_length = norm(designed_point1 - designed_point2);
% 二位性质：中心轴线的高度
designed_height = designed_center(3);
% 二维性质：拟合中心轴线
% raw_line = zeros(1,3);
designed_X = [designed_point1(:,1) designed_point2(:,1)]';
designed_Y = [designed_point1(:,2) designed_point2(:,2)]';
designed_Z = [designed_point1(:,3) designed_point2(:,3)]';
% 求斜率
designed_k = tgent(designed_point1(1),designed_point1(2),designed_point2(1),designed_point2(2));

%% 对竣工模型点云进行拟合
% 读取数据集2
pipe_test = load('a2.txt');
x = pipe_test(:,1);
y = pipe_test(:,2);
z = pipe_test(:,3);
pipe_built = [x,y,z];
% % 点云可视化
% figure(3);
% pcshow([x,y,z]);
% title('pipeBuilt');
% xlabel('X(m)');
% zlabel('Z(m)');
% ylabel('Y(m)');
% 将点云写入pcd格式
designed_pointcloud = pointCloud(pipe_built);
% 将数据写入pcd文件中,pcread需要读取pcd文件
pcwrite(designed_pointcloud,'pipe_built.pcd', 'Encoding', 'ascii');
pipe_built = pcread('pipe_built.pcd');
% 拟合圆柱数据集1
% 设置点到圆柱面的最大距离，目的在于降低噪点对拟合圆柱的影响
maxDistance = 1;
% 执行圆柱体拟合
cylinderModel_built = pcfitcylinder(pipe_built,maxDistance);
% % 画出拟合的结果
% figure(4)
% pcshow(piep_built.Location)
% title('圆柱2拟合效果')
% xlabel('X(m)');
% ylabel('Y(m)');
% zlabel('Z(m)');
% hold on
% plot(cylinderModel_built)

%% 提取竣工模型圆柱拟合的参数
% 半径
built_redius = cylinderModel_built.Radius;
% 圆柱中心点
built_center = cylinderModel_built.Center;
% 中心轴线的长度
built_point1 = cylinderModel_built.Parameters(:,1:3);
built_point2 = cylinderModel_built.Parameters(:,4:6);
built_length = norm(built_point1 - built_point2);
% 二位性质：中心轴线的高度
built_height = built_center(3);
% 拟合中心轴线
% raw_line = zeros(1,3);
built_X = [built_point1(:,1) built_point2(:,1)]';
built_Y = [built_point1(:,2) built_point2(:,2)]';
built_Z = [built_point1(:,3) built_point2(:,3)]';
% 求斜率
built_k = tgent(built_point1(1),built_point1(2),built_point2(1),built_point2(2));

%% 进行比较，需要考虑比较的顺序
% 先比较两点之间的距离，d为设计圆柱中心点画球的一个阈值
ransac_d = 20;%为RANSAC的拟合误差
sphere_r = ransac_d + designed_redius;
% 两点距离
point_distance = norm(built_center - designed_center);
% 求两根直线之间的夹角
x1 = designed_point1(1);
y1 = designed_point1(2);
z1 = designed_point1(3);
x2= designed_point2(1);
y2= designed_point2(2);
z2 = designed_point2(3);
x3 = built_point1(1);
y3 = built_point1(2);
z3 = built_point1(3);
x4 = built_point2(1);
y4 = built_point2(2);
z4 = built_point2(3);
theta = acosd(dot([x1-x2,y1-y2,z2-z1],[x3-x4,y3-y4,z4-z3])/(norm([x1-x2,y1-y2,z2-z1])*norm([x3-x4,y3-y4,z4-z3])));

if point_distance <= sphere_r
    disp('不需要变更')
else
    disp('需要变更')
    % 进行依次对比，并对有问题的进行计数
    % 比较半径
    if abs(designed_redius - built_redius) < sphere_r
        disp('半径没问题')
    else
        disp('半径有问题')
    end
    % 比较高度
    if abs(designed_height - built_height) < sphere_r
        disp('高程没问题')
    else
        disp('高程有问题')
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
        disp('长度没问题')
    else
        disp('长度有问题')
    end
    % 比较方向
    if theta < 5
        disp('角度没问题')
    else
        disp('角度有问题')
    end
end

%% 对结果进行画图
figure
grid on
plot(cylinderModel_designed)
hold on
plot(cylinderModel_built)
hold on
axis equal
% 先对两根圆管的中心轴线进行画图
figure
grid on
axis equal
line(designed_X,designed_Y,designed_Z,'linestyle',':','color','r'); 
hold on
axis equal
line(built_X,built_Y,built_Z,'linestyle','-.','color','g');
% 以设计圆柱中心画球
figure
grid on
[sphere_x,sphere_y,sphere_z]  = ellipsoid(designed_center(1),designed_center(2),designed_center(3),sphere_r,sphere_r,sphere_r);
surf(sphere_x,sphere_y,sphere_z,'FaceAlpha',0.1);
designed_line = line(designed_X,designed_Y,designed_Z,'linestyle',':','color','r','LineWidth',3);
axis equal