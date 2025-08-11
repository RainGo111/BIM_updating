clc;
clear;
close all;
% -------------加载点云数据-----------------
ptCloud_ceiling = pcread('originalroom(0.01).pcd');
% 检测天花板点云，距离以米为单位
maxDistance = 0.05;
referenceVector = [0 0 1];
maxAngularDistance = 5;
[model,inlier_Idx,outlier_Idx] = pcfitplane(ptCloud_ceiling, ...
    maxDistance,referenceVector,maxAngularDistance);
% 提取拟合平面点云
ceiling_inlier = select(ptCloud_ceiling,inlier_Idx);
% 选中除地面点云以外的点云
ceiling_outlier = select(ptCloud_ceiling,outlier_Idx);
% % 对点云进行聚类，每个聚类至少包含 10 个点
% minDistance = 2;
% minPoints = 10;
% [labels,numClusters] = pcsegdist(cloud_outlier,minDistance,'NumClusterPoints',minPoints);
% % 删除标签值为0的点
% idxValidPoints = find(labels);
% labelColorIndex = labels(idxValidPoints);
% segmentedPtCloud = select(cloud_outlier,idxValidPoints);
% for num=1:numClusters
%     idxPoints = find(labels==num);         % 根据分类标签查找同一个类里的点
%     segmented = select(ptCloud,outlier_Idx); % 根据同类点索引提取点
%     filename = strcat('WithoutCeiling','.pcd');
%     pcwrite(segmented,filename,'Encoding','binary'); % 保存结果到本地文件夹
% end
% % -------------加载点云数据-----------------
% ptCloud_floor = pcread('WithoutCeiling.pcd');

% 检测剩余点云中的地面，距离以米为单位
maxDistance = 0.05;
referenceVector = [0 0 1];
maxAngularDistance = 5;
[~,inlier_Idx,outlier_Idx] = pcfitplane(ceiling_outlier, ...
    maxDistance,referenceVector,maxAngularDistance);
% 提取拟合平面点云
floor_inlier = select(ceiling_outlier,inlier_Idx);
% 选中除地面点云以外的点云
floor_outlier = select(ceiling_outlier,outlier_Idx);

% 只剩墙面需要移除
data = floor_outlier;
for t = 1:4
%     clearvars  wall_inlier;
    maxDistance = 0.05;
    referenceVector = [0 1 0];
    maxAngularDistance = 90;
    [~,inlier_Idx,outlier_Idx] = pcfitplane(data, ...
        maxDistance,referenceVector,maxAngularDistance);
    % 提取拟合平面点云
    wall_inlier = select(data,inlier_Idx);
    % 选中除地面点云以外的点云
    wall_outlier = select(data,outlier_Idx);
    % 可视化
    figure
    pcshow(ptCloud_ceiling)
    hold on
    pcshow(ceiling_inlier.Location,[0,1,0])
    hold on
    pcshow(floor_inlier.Location,[1,0,0])
    hold on
    pcshow(wall_inlier.Location,[1,0.5,0])
    hold on
    set(gcf,'color','w')
    set(gca,'color','w')
    set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
    xlabel('X(m)');
    ylabel('Y(m)');
    zlabel('Z(m)');
    view(200,10)
    hold off
    % 剩余点云
    data = wall_outlier;
end

% Export point cloud data
filename = sprintf('without_wall_%d.pcd', t);  % Creates unique filename for each iteration
% Save final point cloud data (after the for loop)
pcwrite(data, 'final_pointcloud.pcd', 'Encoding', 'binary');  % Saves the point cloud in PCD format

% 可视化删除平面
figure
pcshow(data)
set(gcf,'color','w')
set(gca,'color','w')
set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
xlabel('X(m)');
ylabel('Y(m)');
zlabel('Z(m)');
view(200,10)
axis auto
% 保存图片
picture_name=[num2str('figure(2)'),'.jpg'];
saveas(gca,picture_name);
hold on;



% % 可视化平面检测结果
% figure
% pcshow(ptCloud_ceiling)
% hold on
% pcshow(ceiling_inlier.Location,[1,0,0])
% hold on
% pcshow(floor_inlier.Location,[0,1,0])
% hold on
% pcshow(wall_inlier.Location,[0,0,1])
% set(gcf,'color','w')
% set(gca,'color','w')
% set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
% xlabel('X(m)');
% ylabel('Y(m)');
% zlabel('Z(m)');
% view(200,10)
% axis auto
% % 保存图片
% picture_name=[num2str('figure(1)'),'.jpg'];
% saveas(gca,picture_name);
% hold off

% % 可视化删除平面
% figure(2)
% pcshow(wall_outlier)
% set(gcf,'color','w')
% set(gca,'color','w')
% set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
% xlabel('X(m)');
% ylabel('Y(m)');
% zlabel('Z(m)');
% view(200,10)
% axis auto
% % 保存图片
% picture_name=[num2str('figure(2)'),'.jpg'];
% saveas(gca,picture_name);
% hold on;
