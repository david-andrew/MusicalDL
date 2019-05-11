function cents = interval(p0, p1)
%get_interval returns the interval between two pitches, in cents
%   p0 is the reference pitch in Hz
%   p1 is the target pitch in Hz
%   cents is the number of cents from p0 to p1. 1 cent = 1/100 semitone
    cents = (1200/log(2)) .* log(p1 ./ p0);
end

