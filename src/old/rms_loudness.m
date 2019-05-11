function [level, loc] = rms_loudness(sound, FS)
%Compute an analog of the loudness of the signal using RMS
%   sound is the input sound file
%   FS is the sample rate of the sound
%
%   level is the RMS sound level
%   loc is the location of each volume measurement (in samples)

    WindowSize = floor(FS * 0.25);
    OverlapLength = floor(FS * 0.125);
    HopSize = WindowSize - OverlapLength;
    
    %arrays to hold the pitches we find
    num_frames = floor((length(sound) - WindowSize) / HopSize)-1;

    loc = zeros(num_frames, 1);    %array of pitch locations
    level = zeros(num_frames, 1);     %array of pitch values
    
    for frame = 1:num_frames
        index = (frame - 1) * HopSize + 1; %current start of the sample being read

        %root mean square of a window of the sound sample
        sample = sound(index:index+WindowSize-1);
        level(frame) = rms(sample); %sqrt(mean(sample.^2));
        loc(frame) = index;
    end
end




