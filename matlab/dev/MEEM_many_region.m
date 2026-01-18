% Many-region, 2-body matched eigenfunction expansion method
% Rebecca McCabe 12-14-2023

clc;close all

%% 1: Define geometry with parameterized curve
r_b1 = [0:3 2 1];
z_b1 = -5:0;

r_b2 = 4:8;
z_b2 = [0 -1 -2 -1 0];

h = 10;
w = 1;

%% 2: Discretize into fluid zones, with areas of highest curvature getting 
% better resolution (zones with smaller radial thickness)

assert(length(r_b1) == length(z_b1));
n_points_1 = length(r_b1);
assert(length(r_b2) == length(z_b2));
n_points_2 = length(r_b2);

r_all = [r_b1, r_b2];
z_all = [z_b1, z_b2];
assert(all(z_all <= 0)); % cannot extend out of the water
assert(all(z_all >= -h)); % cannot extend below the sea floor
assert(any(z_b1==0)); % must be surface piercing
assert(any(z_b2==0));

figure
plot(r_b1,z_b1,'ro--',r_b2,z_b2,'bo--')
hold on

% for plotting
[r_b1_plot, z_b1_plot, r_b1_discrete, z_b1_discrete] = points_to_plot_points(r_b1,z_b1);
[r_b2_plot, z_b2_plot, r_b2_discrete, z_b2_discrete] = points_to_plot_points(r_b2,z_b2);

r_all_discrete = [r_b1_discrete, r_b2_discrete];
z_all_discrete = [z_b1_discrete, z_b2_discrete];
n_body_discretizations = length(z_all_discrete);

plot(r_b1_plot,z_b1_plot,'r*-',r_b2_plot,z_b2_plot,'b*-')

r_max = 1.2 * max(r_all);
plot([0,r_max],[-h,-h],'k')
plot([0 0],[-h 0],'k--')
wavelength = 2*pi/w;
x = 0:wavelength/20:r_max;
plot(x,sin(w*x),'c')
plot(x,0*x,'c')
legend('body 1','body 2')
xlabel('R')
ylabel('Z')
improvePlot

% Check points to ensure the two bodies don't overlap and are otherwise valid

% Find the type of each fluid zone and assign it the corresponding potential function
% for each point, check if it's the smallest or biggest z for that radius

figure
hold on

for i=1:n_body_discretizations
    r_i = r_all_discrete(i);
    z_i = z_all_discrete(i);
    z_this_r = z_all_discrete(r_all_discrete==r_i);
    
    if i < n_points_1
        body = 1;
    else
        body = 2;
    end

    % innermost region
    r_corners = r_all([i+body-1, i+body]);
    if any(r_corners == 0)
        superscript = '^0';
    else
        superscript = '';
    end

    % detect type of zone
    if z_i == min(z_this_r)
        zone = 'P';
        z_bounds = [-h,z_i];
    elseif z_i == max(z_this_r)
        zone = 'L';
        z_bounds = [z_i,0];
    else
        zone = 'B';
        z_bounds = NaN;
    end

    full_zone = [zone '_' num2str(body) superscript];
    text(r_i,mean(z_bounds),full_zone)
    plot(r_corners(1)*[1,1],z_bounds,'k--')
    plot(r_corners(2)*[1,1],z_bounds,'k--')
end

% intermediate M region
if max(r_b1) < min(r_b2)
    mid_r = mean([max(r_b1), min(r_b2)]);
elseif min(r_b1) > max(r_b2)
    mid_r = mean([min(r_b1), max(r_b2)]);
else
    mid_r = NaN;
end
text(mid_r,-h/2,'M')

% exterior region
text(max(r_all)*1.1,-h/2,'E')

plot(r_b1_plot,z_b1_plot,'r*-',r_b2_plot,z_b2_plot,'b*-')
hold on
r_max = 1.2 * max([r_b1,r_b2]);
plot([0,r_max],[-h,-h],'k')
plot([0 0],[-h 0],'k--')
wavelength = 2*pi/w;
x = 0:wavelength/20:r_max;
plot(x,sin(w*x),'c')
plot(x,0*x,'c')
xlabel('R')
ylabel('Z')
improvePlot

% Check for adjacency to determine which zones should be matched

% Apply potential matching and velocity matching on all adjacent zones

% Apply the zero radial velocity condition and add the resulting equations to the velocity matching equations

% Apply orthogonality and arrive at the symbolic matrix equation

% Plug in the velocity conditions associated with which body is moving 

% Plug in numbers for the specific geometric dimensions

% Solve the matrix equation to find the unknown Fourier coefficients

% Plug the Fourier coefficients into the expression for potential and hydrodynamic coefficients

% Repeat steps 10-12 for different geometries

%% functions
function [r_plot,z_plot,r_discrete,z_discrete] = points_to_plot_points(r,z)
    r_discrete = mean([r(1:end-1);r(2:end)]);
    z_discrete = mean([z(1:end-1);z(2:end)]);

    r_plot = [r(1), repelem(r(2:end),2)];
    z_plot = [repelem(z_discrete(1:end),2) z(end)];
end