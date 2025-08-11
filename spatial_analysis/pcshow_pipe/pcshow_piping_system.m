% ---------------读入点云-------------------
pc = pcread('pipingsystem.pcd');
% --------------可视化点云------------------
figure
pcshow(pc.Location,[1,0,1]);
set(gcf,'color','w')
set(gca,'color','w')
set(gca, 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'ZColor', [0.15 0.15 0.15])
title('Piping System','Color','r','FontSize',16);
xlabel('X(m)');
ylabel('Y(m)');
zlabel('Z(m)');
axis auto
% 保存图片
picture_name=[num2str('Piping System'),'.jpg'];
saveas(gca,picture_name);