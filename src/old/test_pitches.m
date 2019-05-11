clc
clear
close all

[C3, FS]  = audioread('semitone_scale/130.8.wav');
Db3 = audioread('semitone_scale/138.6.wav');
D3  = audioread('semitone_scale/146.8.wav');
Eb3 = audioread('semitone_scale/155.6.wav');
E3  = audioread('semitone_scale/164.8.wav');
F3  = audioread('semitone_scale/174.6.wav');
Gb3 = audioread('semitone_scale/185.0.wav');
G3  = audioread('semitone_scale/196.0.wav');
Ab3 = audioread('semitone_scale/207.7.wav');
A3  = audioread('semitone_scale/220.0.wav');
Bb3 = audioread('semitone_scale/233.1.wav');
B3  = audioread('semitone_scale/246.9.wav');
C4  = audioread('semitone_scale/261.6.wav');

%easy access to the n'th scale degree
scale = {C3, Db3, D3, Eb3, E3, F3, Gb3, G3, Ab3, A3, Bb3, B3, C4};
truth = [130.8 138.6 146.8 155.6 164.8 174.6 185.0 196.0 207.7 220.0 233.1 246.9 261.6]; 

%play a chord (Cmaj add 2 add 7)
% play_chord({C3, D3, E3, G3, B3}, FS);

for i = 1:2%length(scale)
    %t = truth(i);
    [truth, loc_truth] = pitch(scale{i}, FS, 'Method', 'NCF', 'MedianFilterLength', 25);  %ground truth from matlab pitch estimator
    %[bsac, loc_bsac]   = BSAC_pitch_slow(scale{i}, FS);                                 %slow version of our algorithm's prediction
    %[bsac, loc_bsac]   = BSAC_pitch(scale{i}, FS);                                        %our algorithm's prediction
    

    
%     avg_bsac = median(bsac);
%     avg_truth = median(truth);  
%     avg_error = interval(avg_truth, avg_bsac);                                     %pitch error in cents (i.e. hundredths of a semitone)

%     %get the differenc in pitch between truth and bsac at every instant (using interpolation if they don't line up)
%     if loc_truth(end) < loc_bsac(end)
%         error = interval(truth, interp1(loc_bsac, bsac, loc_truth));                      
%     else
%         error = interval(interp1(loc_truth, truth, loc_bsac), bsac);
%     end
%     avg_error = mean(error(~isnan(error)));

%     fprintf('Prediction: %4.2f  |  Actual: %4.2f  |  Error: %8.2f cents\n', avg_bsac, avg_truth, avg_error);   %print results
    
%     figure; hold on
%     plot(loc_bsac/FS, bsac);
%     plot(loc_truth/FS, truth);
%     title('matlab vs bsac');
%     xlabel('time (s)');
%     ylabel('pitch (Hz)');
%     legend('BSAC', 'MATLAB');
%     
%     figure
%     [phon, loc] = rms_loudness(scale{i}, FS);
%     plot(loc/FS, phon);
%     title('Volume');
%     xlabel('time (s)');
%     ylabel('RMS loudness');
    
%     waitforbuttonpress
%     close all
end


