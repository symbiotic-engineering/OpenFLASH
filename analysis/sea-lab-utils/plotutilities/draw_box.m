function [box] = draw_box(fig,location,h,w,angle,c,LW,LS)
% This function draws a box around the location. If no w is provided it is
% assumed square. If no angle is is assumed 0. Angle should be in radians.
figure(fig)
if nargin < 4
    w = h;
end
if nargin < 5
    angle = 0;
end
if nargin < 6
    c = 'k';
end
if nargin < 7
    LW = 1;
end
if nargin < 8
    LS = '-';
end
xc = location(1);
yc = location(2);

d = sqrt((h/2).^2 + (w/2).^2);
x = zeros(4,1);
y = zeros(4,1);
a = atan(h/w);

x(1) = xc + d*cos(a + angle);
y(1) = yc + d*sin(a + angle);
x(2) = xc + d*cos(angle + pi - a);
y(2) = yc + d*sin(angle + pi - a);
x(3) = xc + d*cos(a + angle + pi);
y(3) = yc + d*sin(a + angle + pi);
x(4) = xc + d*cos(angle - a);
y(4) = yc + d*sin(angle - a);

for i = 1:3
    box(i) = line([x(i) x(i+1)],[y(i) y(i+1)],'color',c,'linewidth',LW,'linestyle',LS);
end
box(4) = line([x(1) x(4)],[y(1) y(4)],'color',c,'linewidth',LW,'linestyle',LS);