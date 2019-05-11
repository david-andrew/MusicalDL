function [] = play_chord(chord, FS)
%play_chord() Play the chord by combining the samples
%   chord is a cell array of sound samples
%   FS is the playback frequency of the sounds

min_length = 99999999999;
for i = 1:length(chord)
    if length(chord{i}) < min_length
        min_length = length(chord{i});
    end
end

%pad each note so that it is the same length
chord_sound = zeros(min_length, 1);
for i = 1:length(chord)
    chord_sound = chord_sound + (chord{i}(1:min_length));
end
soundsc(chord_sound, FS);

end

