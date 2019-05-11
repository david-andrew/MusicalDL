% %% Analyze voice harmonics
% 
% [X,FS] = audioread('voice_tone_low.wav');
% L = length(X);
% f = FS*(0:(L/2))/L;
% 
% Y = fft(X);
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% 
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of S(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')

%% Analyze harmonics version 3
close all
clc
clear



[X,FS] = audioread('ooh_tone_low.wav');
L = length(X);
f = FS*(0:(L/2))/L;

Y = fft(X);
Pyy = sqrt(Y.*conj(Y))/L;
P2 = Pyy;
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

% [PKS, LOCS] = findpeaks(P1, 'MinPeakProminence', .005);
[~,loc] = max(P1);
[PKS, LOCS] = findpeaks(P1, 'MinPeakDistance', loc/2, 'MinPeakHeight', 0.001);
[PKS, I] = sort(PKS);
LOCS = LOCS(I);

num_peaks = min(length(PKS),10); %take the 10 highest peaks in the signal

figure
plot(f,P1);
hold on
plot(FS/L*LOCS, PKS, 'o')

chords = {
    [0 4 7 11]; 
    [0 3 7 10];
    };
durations = [
    4;
    4;
    ];

while true
    for k = 1:length(chords)
        chord = chords{k};
        duration = durations(k); %seconds
        tone = zeros(FS*duration,1);%generate 1 second of tone

        for j = 1:length(chord)
            for i = 1:num_peaks %length(PKS)
                f0 = LOCS(i)*FS/L * (2^(1/12))^chord(j);
                a = PKS(i);
                tone = tone + a * sin((1:FS*duration)'*f0*2*pi/FS);
            end
        end
        soundsc(tone,FS);
        pause(duration)
    end
end
% %% playback harmonics
% F = [103.1      206.1   309.3   417.1   521.6       626.1       719.7   822.7       930.4   1031        1132]';
% A = [0.05388    0.03304 0.3556  0.01198 0.002581    0.002739    0.00322 0.001565    1e-4    7.821e-5    2.5e-5]';
