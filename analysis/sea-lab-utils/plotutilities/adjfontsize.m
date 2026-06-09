% Matlab script to adjust the fontsize of the xlabel, ylabel, and axes tic
% labels for the active figure.  The variable "fsize" must be defined in the
% calling script. Updated: 3/17/2014 -PTK

% hAnnotation = get(gcf,'Annotation');
% hLegendEntry = get(hAnnotation','LegendInformation');
% set(hLegendEntry,'FontSize',fsize-2)

set(gca,'FontSize',fsize); %Takes care of tick labels

%The following lines grab annotation, title, and axis label handles
hText = findall(findobj(gcf),'Type','text'); 
set(hText,'FontSize',fsize);
alldatacursors = findall(gcf,'type','hggroup');
set(alldatacursors,'FontSize',fsize);

% These commands aren't needed with the newly implemented "findall" command
% set(get(gca,'Ylabel'),'FontSize',fsize);
% set(get(gca,'Xlabel'),'FontSize',fsize);
% set(get(gca,'Title'),'FontSize',fsize);