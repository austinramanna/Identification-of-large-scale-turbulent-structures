close all
clear all
% Define the number of slices and maximum number of contours
num_slices = 1;  % Adjust as necessary
max_num_contours = 115;  % Adjust to your actual maximum number of contours
all_pca_results = [];
%all_minkowski_results = [];

% Loop over all slices
for slice = 0:num_slices-1

    % Loop over all contours
    for contour =0:max_num_contours
        % Initialize arrays to hold features and their sizes for each contour
        featureN = cell(1, max_num_contours);
        NumpointsN = zeros(1, max_num_contours);
        
        % Generate filename
        filename = sprintf('Merged_Grid_%d.mat', contour);
        
        % Check if file exists
        if isfile(filename)
            
            % Load the data
            data = load(filename);
            gridx = data.grid;
            
            %factor 16 subsampling
            gridx = gridx(1:16:end, 1:16:end, 1:16:end);

            % Define the ticks for the plots
            fs = 10;
            maxtick=550;
            xtick = 0:50:maxtick;  % Or whatever value you want for the x-axis ticks
            ytick = 0:50:maxtick;  % Or whatever value you want for the y-axis ticks
            ztick = 0:10:maxtick;   % Or whatever value you want for the z-axis ticks
            
            % Define range based on data dimensions
            xmin = 1;
            xmax = size(gridx, 1);
            ymin = 1;
            ymax = size(gridx, 2);
            zmin = 1;
            zmax = size(gridx, 3);
            
            % Define range
            rangex = [xmin xmax ymin ymax zmin zmax];
            
            % Define the level for the isosurface
            levelQ = 0.5;
            
            % Generate the isosurface
            dum = isosurface(gridx, levelQ);
            
            % Create a patch object
            pz = patch(dum);
            set(pz, 'FaceColor', 'green', 'EdgeColor', 'none');

            % Set the labels
            xlabel('x [mm]');
            ylabel('y [mm]');
            zlabel('z [mm]');
            daspect([1 1 1]);
            axis(rangex);
            view(3);
            camlight(0, 110-90);
            lighting phong;
            grid on;
            box on;
            view(0, 90);
            
            aa=dum.vertices;%points (x,y,z)
            bb=dum.faces;%triangles defines by points in aa (N1,N2,N3)
            
            
            
            %%
            clear pointsN featureN
            %start from point 1
            istart=1;
            
            %procedure leaves aa unaffected, but
            %bb (faces) are grouped to individual structures
            bb_avail=bb;%faces not yet assigned to a structure
            for N=1:size(aa,1)
                N;
                if mod(N, 1000) == 0
                    fprintf('Iteration %d of %d done...\n', N, size(aa,1));
                end
                bb_struc=[];%faces assigned to structure
            
                %initial point
                gg=istart;
                dlgg=gg;
            
                while ~isempty(dlgg)%new points are added to list gg
            
                    gg_old=gg;
            
                    %find faces corresponding to new points dlgg
                    iface=[];
                    for n=1:length(dlgg)
                        [ii,j]=find(bb_avail==dlgg(n));
                        iface=[iface;ii];
                    end
                    %[iface,j]=find(bb_avail==dlgg);
                    %add faces to structure
                    bb_struc=[bb_struc ; bb_avail(iface,:)];
                    %update available faces (remove faces of structure)
                    bb_avail = setdiff(bb_avail, bb_struc, 'rows');
            
                    %all unique points in structure bb_struc
                    gg=unique(bb_struc);
            
                    %select newly added points (=remove old points from gg)
                    dlgg = setdiff(gg,gg_old);
            
                    clear ifaces ii j
                end
            
            
            
                pointsN{N}=gg(:)';
                NumpointsN(N)=length(gg);
                featureN{N}=bb_struc;%triangles defines by points in aa corresponding to single feature
                

                %new starting point
                usedpoints=sort(cell2mat(pointsN));
                idum=find(gradient(usedpoints)>1);
                if isempty(idum)
                    %istart=gg(end)+1;
                    istart=usedpoints(end)+1;
                else
                    istart=idum(1)+1;
                end
                clear idum
            
                if istart >= size(aa,1)%stop finding new features
                    disp(['done found ' num2str(N) ' features'])
                    break
                end            
            end
               
            
            %filter small structures
            thres_points=100;
            %[NpointsSORT,Isort] = sort(NumpointsN,'descend');
            
            % Get indices of structures with enough points
            enough_points_indices = find(NumpointsN >= thres_points);
            
            % If no structure met the threshold, print the filename
            if isempty(enough_points_indices)
                disp(['No structures met the threshold in file: ', filename]);
                continue;  % skip the rest of the loop for this file
            end

            % Sort these structures in descending order of size
            [NpointsSORT, Isort] = sort(NumpointsN(enough_points_indices),'descend');
            Nstruct_filt=sum((NpointsSORT>=thres_points));

            % Calculate the number of structures to process
            num_required_structures = round(length(Isort)*0.2);
            
            % Initialize the results
            res = zeros(7, num_required_structures);
            pca_res = zeros(6, num_required_structures);
            
            % Define the size of the grid for binarization
            grid_size = size(gridx);

            %%
            for i = 1:Nstruct_filt*0.2
                % Retrieve the structure
                structure_faces = featureN{enough_points_indices(Isort(i))};
                
                % Convert face indices to actual vertices
                structure_vertices = aa(structure_faces, :);
                % Assuming structure_vertices is your segmented 3D structure's vertices
                mean_structure = mean(structure_vertices);
                centered_structure = bsxfun(@minus, structure_vertices, mean_structure);
                
                % Compute PCA
                [coeff,score,latent] = pca(centered_structure);
                
                % The length scales (or standard deviations) are the square roots of the eigenvalues
                standard_deviations = sqrt(latent)';
                
                clear range

                % Compute the range (max-min) along each principal component
                pca_ranges = [range(score(:,1)), range(score(:,2)), range(score(:,3))];
            
                % Inside your loop, after computing PCA results for each structure:

                % Compute PCA length scales and standard deviations
                pca_ranges = [max(score(:,1)) - min(score(:,1)), max(score(:,2)) - min(score(:,2)), max(score(:,3)) - min(score(:,3))];
                pca_std_dev = std(score);
                
                % Append the current structure's results to the main results array
                %all_pca_results = [all_pca_results; pca_ranges, pca_std_dev];
                pca_res(:,i) = [pca_ranges, pca_std_dev];

                % Calculate the Alpha shape
                alpha_Shape = alphaShape(structure_vertices);
                
                % Binarize the alpha shape
                structure_img = false(grid_size);  % Start with all zeros
                [X, Y, Z] = ndgrid(1:grid_size(1), 1:grid_size(2), 1:grid_size(3));
                structure_img(inShape(alpha_Shape, X(:), Y(:), Z(:))) = true;  % Set voxels inside the alpha shape to 1
            
                % Visualize the largest structure
                if i == 1
                     %plot filtered features
                            cm=colormap(hsv(Nstruct_filt));
                            figure
                            hold on
                            for nn=1:Nstruct_filt
                                pz2=patch('Faces',featureN{Isort(nn)},'Vertices',aa);
                                %set(pz2, 'FaceColor', 'green', 'EdgeColor', 'none');
                                set(pz2, 'FaceColor', cm(nn,:), 'EdgeColor', 'none');
                                %
                            end
                            % px=patch(isosurface(x,-y,z,abs(vortz),levelb));
                            % set(px, 'FaceColor', 'blue', 'EdgeColor', 'none');
                            fs = 10;
                            ylab = {'0', '50', '100', '150', '200', '250'};  % Or whatever labels you want for the y-axis ticks
                            xlabel('x/\delta','FontSize',fs)
                            ylabel('z/\delta','FontSize',fs)
                            zlabel('y/\delta','FontSize',fs)
                            set(gca,'FontSize',fs);
                            set(gca,'XTick',xtick);%,'XTickLabel');%
                            set(gca,'YTick',ytick);%,'YTickLabel');
                            set(gca,'ZTick',ztick);%,'ZTickLabel');

                            set(gca,'YTickLabel',ylab);
                            daspect([1 1 1])
                            axis(rangex)
                            view(3)
                            camlight(0,110-90);
                            lighting phong
                            grid on
                            box on
                            view(0,90)
                            figure_name2 = sprintf('largest_structure_Merged_%d.png',  contour);
                            saveas(gcf, figure_name2);
                end
                
                % Calculate Minkowski functionals
                V = sum(structure_img(:));  % Volume
                S = imSurfaceArea(structure_img, [1, 1, 1]);  % Surface Area
                B = imMeanBreadth(structure_img);  % Euler characteristic
                [Chi, ~] = imEuler3d(structure_img);
            
                V0 = V;
                V1 = S / 6;
                V2 = B * 2 / 3;
                V3 = Chi * 2;  % *((4*pi)^2)/(3*2*pi)
            
                T = V0 / (2 * V1);
                W = 2 * V1 / (pi * V2);
            
                if V3 < 0.0000001
                    G = 1 - V3 / 2;
                    L = 3 * V2 / (4 * (G + 1));
                else
                    L = (3 * V2) / (2 * V3);
                end
                % Store the results
                res(:,i) = [V; S; B;V3;T;W;L];
                %minkowski_results{i} = struct('V0', V0, 'V1', V1, 'V2', V2, 'V3', V3, 'T', T, 'W', W, 'L', L);
            end
            % Save the results
            %all_minkowski_results = [all_minkowski_results, res];
            results_filename = sprintf('High_Re_minkowski_results_slice_%d_contour_%d.mat', slice, contour);
            save(results_filename, 'res');

            pca_results_filename = sprintf('PCA_results_slice_%d_contour_%d.mat', slice, contour);
            save(pca_results_filename, 'pca_res');
        end
    end
end
%save('all_minkowski_results.mat', 'all_minkowski_results');
%save('all_pca_results.mat', 'all_pca_results');