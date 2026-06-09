% convert figure to black background
% set(gca,'Color','w')
% set(findobj())
whitebg(gcf,'k')
set(gcf,'Color','k')
set(gcf,'InvertHardcopy','off')
hText = findall(findobj(gcf),'Type','text'); 
set(hText,'Color','w')
set(hText,'BackgroundColor','none')

hLine = findall(findobj(gcf),'Type','line');
for ii=1:length(hLine)
    if get(hLine(ii),'Color') == [0,0,0]
        set(hLine(ii),'Color','w')
    else
%         set(hLine(ii),'Color','g')
    end
%    set(hLine(ii),'MarkerEdgeColor','c')
end