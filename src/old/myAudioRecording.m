% recorder = audiorecorder(16000, 8, 1, 5);
% set(recorder,'TimerPeriod',1,'TimerFcn',{@audioTimer});
% record(recorder);



% Function that records sound from the microphone at a specified sampling
% rate for a fixed number of seconds
%
% Fs           - the sampling rate (Hz)
% durationSecs - the (optional) duration of the recording (seconds)
% N            - the (optional) FFT N-point block size
function [recorder] = myAudioRecording(Fs,durationSecs,N)

    if ~exist('durationSecs','var')
        % default to five minutes of recording
        durationSecs = 300;
    end
    
    if ~exist('N','var')
        % default to the sampling rate
        N = Fs;
    end
    
    % add an extra half-second so that we get the full duration in our
    % processing
    durationSecs = durationSecs + 0.5;
    
    % index of the last sample obtained from our recording
    lastSampleIdx = 0;
    
    % start time of the recording
    atTimSecs     = 0;
    
    % create the audio recorder
    recorder = audiorecorder(Fs,8,1);
    
    % assign a timer function to the recorder
    set(recorder,'TimerPeriod',0.01,'TimerFcn',@audioTimerCallback);
    
    % create a figure with two subplots
    hFig   = figure;
    hFig = plot(NaN, NaN);

    drawnow;
    
    % start the recording
    record(recorder,durationSecs);
    
    % define the timer callback
    function audioTimerCallback(hObject,~)
        
        % get the sample data
        samples  = getaudiodata(hObject);
        
        % skip if not enough data
        if length(samples)<lastSampleIdx+1+Fs
            return;
        end
        
        % extract the samples that we have not performed an FFT on
        X = samples(lastSampleIdx+1:lastSampleIdx+Fs);
        [f0, loc] = pitch(X,Fs);
        
        % plot the data
%         t = linspace(0,1-1/Fs,Fs) + atTimSecs;
        set(hFig,'XData',loc,'YData',f0);
        
%         f = 0:Fs/N:(Fs/N)*(N-1);
%         set(hPlot2,'XData',f,'YData',abs(Y));
%          
        % increment the last sample index
        lastSampleIdx = lastSampleIdx + Fs;
        
        % increment the time in seconds "counter"
        atTimSecs     = atTimSecs + 1; 
    end

    % do not exit function until the figure has been deleted
    waitfor(hFig);
end