function y=tgent(x1,y1,x2,y2)
%求两点连线的斜率
if x1==x2
    disp('error:斜率不存在')
else
    y=(y1-y2)./(x1-x2);
end