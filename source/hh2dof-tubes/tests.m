% parameters = [mass_a, mass_b, omega_x, omega_y, delta];
parameters = [1, 1, 1, 1, 1];
total_energy = 0.17;
alpha = 0.4;
fs = draw_energysurf(parameters, total_energy, alpha);



%%

numpts = 200;
xVec = linspace(-1.5, 1.5, numpts);
yVec = linspace(-1.5, 1.5, numpts);
[xMesh, yMesh] = meshgrid(xVec, yVec);
peMesh = potential_energy(xMesh, yMesh, parameters);

pe_range = -0.35:0.05:0.35;
target_energy = 0.17;

figure()
contour(xMesh, yMesh, peMesh, pe_range)
hold on
contour(xMesh, yMesh, peMesh, [target_energy target_energy], '-k')
colormap parula
colorbar
hold on
% plot(xx(:,1), xx(:,2), '-r','Linewidth', 2)
% plot(xx(1,1), xx(1,2), 'xk')


numpts = 50;
xVec = linspace(-1.5, 1.5, numpts);
yVec = linspace(-1.5, 1.5, numpts);
[xMesh, yMesh] = meshgrid(xVec, yVec);
peMesh = potential_energy(xMesh, yMesh, parameters);

figure()
surf(xMesh, yMesh, peMesh)
shading interp
colormap parula
colorbar

