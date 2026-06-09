function add_arrow_fun(N,S)
% add_arrow_fun(N,S)
% Inserts N arrows with a circular head of size S
% Defaults: N=1, S=3

%Set defaults for N and S
if(~exist('N','var'))  %if headsize was not supplied, default to 3
    N = 1;
end
if(~exist('S','var'))  %if headsize was not supplied, default to 3
    S = 3;
end

for iii=1:N
    hdumdum = annotation('arrow');
    set(hdumdum,'HeadStyle','ellipse')
    set(hdumdum,'HeadSize',S)
end