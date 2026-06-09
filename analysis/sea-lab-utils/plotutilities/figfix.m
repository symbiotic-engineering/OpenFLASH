function [fwidth, fheight] = figfix(style,fsize,s)
% figfix
% This function adjusts the font size and figure size
% 'style': string defining the figure style
%       'Pap1'   - single column print with W=H
%       'Print1' - single column print with W=4/3H
%       'Print2' - 2 column print out with W = 5/2H
%       'Gold'

% 'fsize': fontsize (number) in pts
set(gca,'ActivePositionProperty','Outerposition');

if exist('fsize','var')==0
    fsize = 20;
elseif exist('style','var')==1
    %the fontsize for the plot was defined in the function call
end

if exist('style','var')==0
    style  = 'ME373_L20';  %if no style is applied, set default
elseif exist('style','var')==1
    %the style for the plot was defined in the function call
else
    disp('Is style defined or not???')
    return
end

switch style
    case 'square'
        fwidth = 4;
        fheight = 4;
    case 'Pap1' %slightly taller figure than "Print1"
        % Single column paper settings
        fwidth = 3.25*s;     %This is the figure width in inches
        fheight = fwidth;
    case 'Print1'
        % Standard Printout
        fwidth = 3.25;
        fheight = .75*fwidth;
    case 'Print2'
        % Wide Printout
        fwidth = 6.5;
        fheight = 0.4*fwidth;   %This is the figure height in inches
    case 'Print3'
        % Standard Printout
        fwidth = 2.9;
        fheight = 1.49;
    case 'Print3' %slightly widerfigure than "Print1"
        % Single column paper settings
        fwidth = 4.25*s;     %This is the figure width in inches
        fheight = 3.25;
    case 'Print4' %Single Column Tall
        fsize = 10;
        fwidth = 3.25 *s;
        fheight = 1.1 * fwidth;
    case 'Gold'
        % Golden ratio
        % fheight = 2/(1+sqrt(5))*fwidth;   %This is the figure height in inches
    case 'ES15'
        fwidth = 5;
        fheight = .75*fwidth;
    case 'ES16'
        fwidth = 4.;
        fheight = 4.;
    case 'ES_deltas'
        fwidth = 5.7;
        fheight = 5.4;
    case 'Pres1'
        fwidth = 4.5;
        fheight = 5;
    case 'Poster'
        fwidth = 6;
        fheight = .75*fwidth;
    case 'PosterTall'
        fwidth = 5;
        fheight = 5;
    case 'BigPic'
        fwidth = 5.5;
        fheight = 6;
    case {'ME373_L20', 'ME373','ME333'}
        fwidth = 10;
        fheight = 6;
    case 'WordCloud'
        fwidth  = 8;
        fheight = 6; 
    
    otherwise
        disp('Select a "style"!')
        return
end
% set(gca,'LineWidth',1)
% set(gca,'XLim',[800 1600])
% set(gca,'YLim',[0 10])
adjfigsize
if strcmp(style,  'WordCloud')
    whitebg(gcf,'k')
    set(gcf,'Color','none')
    set(gcf,'InvertHardcopy','off')
    hText = findall(findobj(gcf),'Type','text'); 
    set(hText,'BackgroundColor','none')
else

end
adjfontsize
box on
movegui('onscreen')
return