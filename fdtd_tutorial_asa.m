%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Copyright 2021 Brian Hamilton                                                                                   %
%                                                                                                                  %
%  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated    %
%  documentation files (the "Software"), to deal in the Software without restriction, including without            %
%  limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of       %
%  the Software, and to permit persons to whom the Software is furnished to do so, subject to the following        %
%  conditions:                                                                                                     %
%                                                                                                                  %
%  The above copyright notice and this permission notice shall be included in all copies or substantial            %
%  portions of the Software.                                                                                       %
%                                                                                                                  %
%  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT           %
%  LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO       %
%  EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN    %
%  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE    %
%  OR OTHER DEALINGS IN THE SOFTWARE.                                                                              %
%                                                                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                                                  %
%  FDTD tutorial for 180th ASA meeting - Matlab code                                                               %
%                                                                                                                  %
%  Simulates 2D wave equation with wall absorption in a rectangular room with circular dome                        %
%                                                                                                                  %
%  Depending on your version of Matlab and Nvidia GPU may need to install the CUDA toolkit and recompile .cu file  %
%                                                                                                                  %
%  For GPU execution you will need a Nvidia GPU (Kepler architecture+) and the Parallel Computing Toolbox          %
%  See: https://uk.mathworks.com/help/parallel-computing/gpu-support-by-release.html                               %
%                                                                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;

c = 343; %speed of sound m/s (20degC)
fmax = 1000; %Hz
PPW = 6; %points per wavelength at fmax
duration = 0.1; %seconds
refl_coeff = 0.9; %reflection coefficient

Bx = 10; By = 4; %box dims (with lower corner at origin)
x_in = Bx*0.5; y_in = By*0.5; %source input position 
R_dome = By*0.5; %heigh of dome (to be centered on roof of box)

draw = true; %to plot or not
add_dome = true; %add dome to scene
apply_rigid = true; %apply rigid boundaries
apply_loss = true; %apply loss 
use_gpu = false;  %for GPU processing (with a supported Nvidia GPU)
recompile_ptx = false; %force recompile of CUDA kernels

assert(R_dome<=0.5*By);
%calculate scene bounds
if add_dome
   Lx = Bx;
   Ly = By+R_dome;
else
   Lx = Bx;
   Ly = By;
end

if (apply_loss)
   assert(apply_rigid);
end

%calculate grid spacing, time step, sample rate
dx = c/fmax/PPW; %grid spacing
dt = sqrt(0.5)*dx/c;
SR = 1/dt;
fprintf('sample rate = %.3f Hz\n',SR) 
fprintf('Δx = %.5f m \n',dx) 

Nx = ceil(Lx/dx)+2; %number of points in x-dir
Ny = ceil(Ly/dx)+2; %number of points in y-dir
Nt = ceil(duration/dt); %number of time-steps to compute

xv = [0:Nx-1]*dx-0.5*dx; %x-sampling points
yv = [0:Ny-1]*dx-0.5*dx; %y-sampling points
[X,Y] = ndgrid(xv,yv);

in_mask = false(Nx,Ny); %mask for 'interior' points
in_mask(X(:)>=0 & Y(:)>=0 & X(:)<Bx & Y(:)<By) = true;
if add_dome
   in_mask((X(:)-0.5*Bx).^2+ (Y(:)-By).^2<R_dome^2) = true;
end
clear X Y;

if (apply_rigid)
   %calculate number of interior neighbours (for interior points only)
   K_map = zeros(Nx,Ny);
   K_map(2:Nx-1,2:Ny-1) = K_map(2:Nx-1,2:Ny-1) + in_mask(3:Nx,2:Ny-1);
   K_map(2:Nx-1,2:Ny-1) = K_map(2:Nx-1,2:Ny-1) + in_mask(1:Nx-2,2:Ny-1);
   K_map(2:Nx-1,2:Ny-1) = K_map(2:Nx-1,2:Ny-1) + in_mask(2:Nx-1,3:Ny);
   K_map(2:Nx-1,2:Ny-1) = K_map(2:Nx-1,2:Ny-1) + in_mask(2:Nx-1,1:Ny-2);
   K_map(~in_mask) = 0;
   ib = find(K_map(:)>0 & K_map(:)<4);
   Kib = K_map(ib);
   clear K_map;
end

%initialise state arrays
u0 = zeros(Nx,Ny);
u1 = zeros(Nx,Ny);
u2 = zeros(Nx,Ny);

%set up an excitation signal
u_in = zeros(Nt,1);
Nh = ceil(5*SR/fmax);
u_in(1:Nh) = 0.5-0.5*cos(2*pi*(0:Nh-1)'./Nh);
u_in(1:Nh) = u_in(1:Nh).*sin(2*pi*(0:Nh-1)'./Nh);

%grid forcing points
inx = round(x_in/dx+1.5)+1;
iny = round(y_in/dx+1.5)+1;
assert(in_mask(inx,iny));

if (draw)
   %a mask convenient for plotting
   draw_mask = NaN*in_mask;
   draw_mask(in_mask) = 1;
end

if (apply_loss)
   %calculate specific admittance γ (g)
   assert(abs(refl_coeff)<=1.0);
   g = (1-refl_coeff)/(1+refl_coeff);
   lf = 0.5*sqrt(0.5)*g; %a loss factor
end

%GPU processing: move data to GPU and compile CUDA kernels
if (use_gpu)
   gpuDevice
   %move arrays to GPU 
   u0 = gpuArray(u0);
   u1 = gpuArray(u1);
   u2 = gpuArray(u2);
   in_mask = gpuArray(in_mask);
   if (apply_rigid)
      ib = gpuArray(int32(ib));
      Kib = gpuArray(int32(Kib));
   end
   
   if isempty(dir('./kernel_2d.ptx')) || (recompile_ptx)
      err = system(['nvcc -ptx -arch=sm_35 -O3 ','kernel_2d.cu']);
      if (err==0)
         fprintf('compiled kernel successfully\n')
      else
         fprintf('error compiling kernel: code = %d \n',err)
         return
      end
   end

   %thread block and thread-block grid dims
   cuBx=32;
   cuBy=8;
   cuGy = floor((Ny-1)/cuBy)+1;
   cuGx = floor((Nx-1)/cuBx)+1;

   k1 = parallel.gpu.CUDAKernel(['kernel_2d.ptx'], ['kernel_2d.cu'], 'air_update');
   k1.ThreadBlockSize = [cuBx cuBy];
   k1.GridSize = [cuGx cuGy];

   if (apply_rigid)
      cuBb=128;
      Nb = length(ib);
      cuGb = floor((Nb-1)/cuBb)+1;
      k2 = parallel.gpu.CUDAKernel(['kernel_2d.ptx'], ['kernel_2d.cu'], 'rigid_update');
      k2.ThreadBlockSize = [cuBb];
      k2.GridSize = [cuGb];
      if (apply_loss)
         k3 = parallel.gpu.CUDAKernel(['kernel_2d.ptx'], ['kernel_2d.cu'], 'apply_loss');
         k3.ThreadBlockSize = [cuBb];
         k3.GridSize = [cuGb];
      end
   end
end

tt = tic;
bb = 0;
for nt=0:Nt-1
   %fdtd update
   if (use_gpu)
      %matlab calls CUDA kernels
      u0=feval(k1, u0, u1, u2, Nx, Ny, in_mask);
      if (apply_rigid)
         u0=feval(k2, u0, u1, u2, Nx, Nb, ib, Kib);
         if (apply_loss)
            u0=feval(k3, u0, u2, Nx, Nb, ib, Kib, lf);
         end
      end
   else
      %regular matlab
      u0(2:Nx-1,2:Ny-1) = in_mask(2:Nx-1,2:Ny-1).*(0.5*(u1(3:Nx,2:Ny-1) + u1(1:Nx-2,2:Ny-1) + u1(2:Nx-1,3:Ny) + u1(2:Nx-1,1:Ny-2)) - u2(2:Nx-1,2:Ny-1));
      if (apply_rigid)
         u0(ib) = (2-0.5*Kib).*u1(ib) + 0.5*(u1(ib+1) + u1(ib-1) + u1(ib+Nx) + u1(ib-Nx)) - u2(ib);
         if (apply_loss)
            u0(ib) = (u0(ib) + lf*(4-Kib).*u2(ib))./(1+lf.*(4-Kib));
         end
      end
   end

   %inject source
   u0(inx,iny) = u0(inx,iny) + u_in(nt+1);

   %plotting
   if (draw)
      if (use_gpu)
         u1g = gather(u1);
      else
         u1g = u1;
      end
      if nt==0
         figure('name','float');
         u_draw = (u1g.*(draw_mask)).';
         hh = imagesc(xv,yv,u_draw,'cdatamapping','scaled');
         set(gca,'ydir','normal');
         axis equal;
         xlabel('x');
         ylabel('y');
         colorbar;
         colormap(bone);
         xlim([min(xv) max(xv)]);
         ylim([min(yv) max(yv)]);
      else
         umax = max(abs(u1g(:)))+eps;
         u_draw = (u1g.*(draw_mask)).';
         set(hh,'cdata',u_draw);
         caxis([-umax umax]);
         drawnow;
      end
   end

   %step forward in time
   u2 = u1; 
   u1 = u0;

   %print time elapsed and performance in megavoxels/s
   tr=toc(tt);
   pstr = sprintf('Progress: nt=%d out of %d time-steps, %.1f megavox/s, %.2f s elapsed',nt+1,Nt,(nt*Nx*Ny)/tr/1e6,tr);
   fprintf([repmat('\b',[1 bb]),pstr]) %erase current line with backspaces
   bb = length(pstr);
end
%print last samples at source point
fprintf('\nlast samples=\n%s',sprintf('%.15f\n',[u0(inx,iny);u1(inx,iny);u2(inx,iny)]))
