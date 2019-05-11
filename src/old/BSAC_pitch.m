function [f0, loc] = BSAC_pitch(sound, FS)
% BSAC_pitch (BitStreamAutoCorrelation) returns the pitch of a given sound
%   The algorithm is detailed here: https://www.cycfi.com/2018/03/fast-and-efficient-pitch-detection-bitstream-autocorrelation/
%   right now it's not very pretty. maybe add schmidt trigger behavior?


SHOW_PLOT = true;   %whether or not to show the plot of the correlation
bitwidth = 32;      %datatype to be used for storing booleans for xor


%convert the sound into bits (stored as double {0, 1})
bitstream = double(sound >= 0);

%strip off bits until a zero-crossing is on both sides
start = 1;
while bitstream(1) == bitstream(start); start = start + 1; end

%strip off trailing bits that won't fit into a whole datatype value
stop = length(bitstream) - mod((length(bitstream) - start + 1), bitwidth);
bitstream = bitstream(start:stop); 

%pack the bitstream into uint32:
pows = 2.^(0:31);       %used to pack bitstream into uint32
inv_pows = 2.^(-31:0);  %used to "unpack" uint32 to bitstream
num_bytes = length(bitstream) / 32;
bitstream = uint32(pows * reshape(bitstream, 32, num_bytes))';



%set up a reasonable window and overlap length (based on human voice range)
% min_frequency = 16.0; %Hz (C0). In general humans don't sing this low
% min_length = floor(FS / min_frequency); %minimum period length for our lowest note 
% window_size = min_length * 4

%min_frequency = 58.27;     %Hz. Lowest reasonable pitch for humans to sing
%max_frequency = 1396.91;    %Hz. Highest reasonable pitch for humans to sing
min_frequency = 16.35;      %C0 (Hz). Lowest humans can typically hear
max_frequency = 4186.01;    %C8 (Hz). Maximum reasonable pitch for humans to sing
WindowSize = floor(FS / min_frequency * 2.75 / bitwidth);
OverlapLength = floor(WindowSize * 0.25);
HopSize = WindowSize - OverlapLength;

% WindowSize = floor(FS * 0.25 / bitwidth);     %in terms of uint32, i.e. 32 samples
% OverlapLength = floor(FS * 0.125 / bitwidth);
% HopSize = WindowSize - OverlapLength;

%arrays to hold the pitches we find

num_frames = floor((length(bitstream) - WindowSize) / HopSize) - 1;

loc = zeros(num_frames, 1);    %array of pitch locations
f0 = zeros(num_frames, 1);     %array of pitch values

%while index + WindowSize < length(sound) - WindowSize
%this may be faster as a parfor? Though on my machine it seems slower
for frame = 1:num_frames %count tracks what index of the parallel for loop this is.      
    index = (frame-1) * HopSize + 1; %current start of the sample being read

    %take sample from start and get vector lengths
    sample = bitstream(index:index+WindowSize-1);
    test_length = 2 * WindowSize; %length(sound)-length(sample)+1; %how many correlations are tested
    test = bitstream(index:index+test_length-1);

    %allocate array to hold results of the autocorrelations 
    correlations = zeros((test_length - WindowSize) * bitwidth, 1,  'double'); 

    %perform bitstream auto correlation
    for i = 1:(test_length - WindowSize)
        for bit = 0:bitwidth-1
            %take XOR of sample accross window, and quickly sum all bits
            X = bitxor(sample, bitshift(test(i:i+WindowSize-1), -bit) + bitshift(test(i+1:i+WindowSize), -bit + bitwidth));
            X = rem(floor(double(X) * inv_pows), 2);
            correlations((i-1)*bitwidth+bit+1) = sum(X(:)) / WindowSize / bitwidth;
        end
    end
    
    %find the troughs in the signal
    [mins_y, mins_x] = findpeaks(-correlations, 'MinPeakProminence', 0.05);
    mins_y=-mins_y; %reinvert signal back to normal

    %super fast pitch detector with minimal required window size
    %reasonable human notes fall within the range of Bb1 (58.27	Hz) and F6
    %(1396.91 Hz), according to Matt. So we are looking for the lowest peak
    %that is comes after the start, and is in that range.
    %human hearing is from 20 Hz to 20 kHz
    
    
    %remove any peaks before maximum frequency possible
    mins_y = mins_y(mins_x > FS / max_frequency);
    mins_x = mins_x(mins_x > FS / max_frequency);
    
    [y, idx] = min(mins_y);
    for i = 1:idx-1
        if abs(y - mins_y(i)) < 0.01 %i.e. this peak is also in the min range
            idx = i;
            break
        end
    end
    
    
            
    if SHOW_PLOT && frame == 1    %plot minimums vs those selected
        plot((1:length(correlations))/FS, correlations)
        hold on
        plot(mins_x/FS, mins_y, 'o');
        plot(mins_x(idx)/FS, mins_y(idx), 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g');
        plot(mins_x(idx)/FS, mins_y(idx), 'ko', 'MarkerSize', 10);
        plot(mins_x(idx)/FS, mins_y(idx), 'kx', 'MarkerSize', 10);
        xlabel('time (s)')
        ylabel('XOR (normalized)');
        title('Auto-Correlation Response');
    end
    
    period_length = mins_x(idx);    

%     %instead of above, perform kmeans clustering on the peaks
%     k = 3; %cluster around up to k means. I think we can actually pump this to be larger?
%     [clusters, means] = kmeans(mins_y, k);
%     sizes = sum((clusters == 1:k), 1); %compute the number of elements in each cluster
% 
%     %find the best cluster (i.e. cluster with more than 5 elements that is minimum)
%     best_cluster = -1;
%     best_mean = inf;
%     for i = 1:k %for each cluster, plot points in that color
%         if sizes(i) >= 5 && means(i) < best_mean %if cluster has enough elements and has the lowest mean (i.e. fundamental frequency highest correleation)
%             best_mean = means(i);
%             best_cluster = i;
%         end      
%     end
% 
% 
%     if SHOW_PLOT && frame == 1    %plot minimums vs those selected
%         figure; hold on
%         colors = 'mrgcbk';
%         plot((1:mins_x(end))/FS, correlations(1:mins_x(end)));
%         for i = 1:k
%             plot(mins_x(clusters==i)/FS, mins_y(clusters==i), 'o', ...
%                 'MarkerEdgeColor', 'k', 'MarkerFaceColor', colors(i));
%             if i == best_cluster
%                 plot(mins_x(clusters==i)/FS, mins_y(clusters==i), 'ko', 'MarkerSize', 10);
%                 plot(mins_x(clusters==i)/FS, mins_y(clusters==i), 'kx', 'MarkerSize', 10);
%             end
%         end
%         xlabel('time (s)')
%         ylabel('XOR (normalized)');
%         title('Auto-Correlation Response');
%     end


    %compute period by looking at the average differenc between each index
%     period_length = median(diff(mins_x(clusters==best_cluster)));%mode(diff(mx));

    pitch = FS / period_length;
    
    %update our list of pitches
    loc(frame) = start + index * bitwidth;
    f0(frame) = pitch;
    
    
end




