function [f0, loc] = BSAC_pitch_slow(sound, FS)
% BSAC_pitch (BitStreamAutoCorrelation) returns the pitch of a given sound
%   The algorithm is detailed here: https://www.cycfi.com/2018/03/fast-and-efficient-pitch-detection-bitstream-autocorrelation/
%   right now it's not very pretty. maybe add schmidt trigger behavior?


SHOW_PLOT = false; %whether or not to show the plot of the correlation

%convert the sound into bits (booleans)
sound = sound >= 0;

%strip off bits until a zero-crossing is on both sides
start = 1; stop = length(sound);
while sound(1) == sound(start)
    start = start + 1;
end
%probably don't actually need to strip trailing crossings
% while sound(end) == sound(stop)
%     stop = stop - 1;
% end
sound = sound(start:stop); %strip off any waveform before first zero cross and after last zero cross

%set up a reasonable window and overlap length (based on human voice range)
% min_frequency = 16.0; %Hz (C0). In general humans don't sing this low
% min_length = floor(FS / min_frequency); %minimum period length for our lowest note 
% window_size = min_length * 4
WindowSize = floor(FS * 0.25);
OverlapLength = floor(FS * 0.125);
HopSize = WindowSize - OverlapLength;

%arrays to hold the pitches we find
num_frames = floor((length(sound) - WindowSize) / HopSize)-1;

loc = zeros(num_frames, 1);    %array of pitch locations
f0 = zeros(num_frames, 1);     %array of pitch values

%count = 1;                      %how many pitches so far
%index = 1;                      %sample location of current pitch
%while index + WindowSize < length(sound) - WindowSize
for frame = 1:num_frames    
    index = (frame - 1) * HopSize + 1; %current start of the sample being read
    
    
    %take sample from start and get vector lengths
    sample = sound(index:index+WindowSize-1);
%     sample_length = length(sample);
    test_length = 2 * WindowSize; %length(sound)-length(sample)+1; %how many correlations are tested
    test = sound(index:index+test_length-1);

    %allocate array to hold results of the autocorrelations 
    correlations = zeros(test_length, 1,  'double'); 

    %perform bitstream crosscorrelation
    parfor i1 = 1:test_length - WindowSize
        correlations(i1) = sum( xor( sample, test(i1:i1+WindowSize-1) ) ) / WindowSize; %normalize
    end



    %find the troughs in the signal
    [mins_y, mins_x] = findpeaks(-correlations, 'MinPeakProminence', 0.05);
    mins_y=-mins_y; %reinvert signal back to normal


    %instead of above, perform kmeans clustering on the peaks
    k = 3; %cluster around up to k means. I think we can actually pump this to be larger?
    [clusters, means] = kmeans(mins_y, k);
    sizes = sum((clusters == 1:k), 1); %compute the number of elements in each cluster

    %find the best cluster (i.e. cluster with more than 5 elements that is minimum)
    best_cluster = -1;
    best_mean = inf;
    for i = 1:k %for each cluster, plot points in that color
        if sizes(i) >= 5 && means(i) < best_mean %if cluster has enough elements and has the lowest mean (i.e. fundamental frequency highest correleation)
            best_mean = means(i);
            best_cluster = i;
        end      
    end


    if SHOW_PLOT && frame == 1    %plot minimums vs those selected
        figure; hold on
        colors = 'mrgcbk';
        plot((1:mins_x(end))/FS, correlations(1:mins_x(end)));
        for i = 1:k
            plot(mins_x(clusters==i)/FS, mins_y(clusters==i), 'o', ...
                'MarkerEdgeColor', 'k', 'MarkerFaceColor', colors(i));
            if i == best_cluster
                plot(mins_x(clusters==i)/FS, mins_y(clusters==i), 'ko', 'MarkerSize', 10);
                plot(mins_x(clusters==i)/FS, mins_y(clusters==i), 'kx', 'MarkerSize', 10);
            end
        end
        xlabel('time (s)')
        ylabel('XOR (normalized)');
        title('Auto-Correlation Response');
    end


    %compute period by looking at the average differenc between each index
    period_length = median(diff(mins_x(clusters==best_cluster)));%mode(diff(mx));

    pitch = FS / period_length;
    
    %update our list of pitches
    loc(frame) = start + index;
    f0(frame) = pitch;
end

