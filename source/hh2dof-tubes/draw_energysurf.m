function fs = draw_energysurf(parameters, H_val, alpha)


    % plot properties
    axesFontName = 'factory';
    % axesFontName = 'Times New Roman';
    axFont = 15;
    textFont = 15;
    labelFont = 20;
    lw = 2;    
    set(0,'Defaulttextinterpreter','latex', ...
        'DefaultAxesFontName', axesFontName, ...
        'DefaultTextFontName', axesFontName, ...
        'DefaultAxesFontSize',axFont, ...
        'DefaultTextFontSize',textFont, ...
        'Defaultuicontrolfontweight','normal', ...
        'Defaulttextfontweight','normal', ...
        'Defaultaxesfontweight','normal');

    esurf = @(x,y,py) (py.^2)/(2*parameters(2)) ...
        + 0.5*parameters(3)^2*x.^2 +  0.5*parameters(4)^2*y.^2 ...
        + x.^2.*y - (parameters(5)/3)*y.^3 ...
        - H_val;
    
%     rgb_col = [51/255 153/255 51/255]; % green
    lightGrey1   = [0.85 0.85 0.85];
    lightGrey2   = [0.7 0.7 0.7];

    darkGrey1  = [0.4 0.4 0.4];
    darkGrey2  = [0.2 0.2 0.2];
    rgb_col = darkGrey1;
    
%     xi = -1.5; xf = 1.5;
    xi = -0.5; xf = 0.5;
%     yi = -1.75; yf = 1.5;
    yi = 0; yf = 1.5;
    pxi = -5; pxf = 5;

    fs = fimplicit3(esurf,[xi xf yi yf pxi pxf],...
        'EdgeColor','none','MeshDensity',100,'FaceAlpha',alpha,...
        'FaceColor',rgb_col);
  
    xlabel('$x$','FontSize',labelFont,'Interpreter','Latex');
    ylabel('$y$','FontSize',labelFont,'Interpreter','Latex');
    zlabel('$p_x$','FontSize',labelFont,'Interpreter','Latex');
    light;
    
    fs.FaceAlpha = alpha;
%     fs.AmbientStrength = 0.8;
    
end





