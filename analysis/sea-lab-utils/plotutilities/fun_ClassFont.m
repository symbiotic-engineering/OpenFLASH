function fun_ClassFont(fsize)
% Enter a font size for the variable 'fsize' to modify the font size for
% display OR enter 'clear' for fsize to reset to the default font size.

s = settings;
if fsize == 'clear'
    clearPersonalValue(s.matlab.fonts.codefont.Size)      
elseif exist('fsize','var')==0
    fsize = 20;
else
    %the fontsize for the plot was defined in the function call
end

s.matlab.fonts.codefont.Size.PersonalValue = fsize; %set the font size

end
% return
%% To clear the personal value for the font size....
