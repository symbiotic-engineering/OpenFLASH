% adjfigsize
% This script adjusts the dimensions of the current figure based on the
% inputs:
% fwidth - This is the desired width of the figure in inches.  Must be
% defined in the calling script

figtouse = gcf;
set(gcf,'Units','inches');
Coord=get(figtouse,'Position');
% fwidth = 3.25;

% Note:
% Coord(1) is your x
% Coord(2) is your y
% Coord(3) is your width
% Coord(4) is your height

% Define modified dimensions
NewWidth = fwidth;
NewHeight = fheight;

% Then set the new handles.

set(figtouse,'Position',[Coord(1),Coord(2),NewWidth,NewHeight]);

%Set Figure Colors
set(gca, 'Color', [1 1 1]); % Sets axes background
set(gcf, 'Color', [1 1 1]); % Sets figure background