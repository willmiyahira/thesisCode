% This code is a sample used for generating a RF ACZ trap on the atom chip

%% define the sweeping parameters
startFreq = 11e6;
midFreq = 19.975e6;
endFreq = 20.08e6;         

sweepTime1 = 85e-3; 
sweepTime2 = 700e-3;

% set the phase difference between signals
phasediff = 295;

% reference frequency (do not change)
refFreq = 100e6;
%% set A/B to be phase shifted
% Establish a connection to the FlexDDS via IP + slot#
[t1, stack1] =openconn('192.168.0.45', 0);
[knownCFR1, stack1] = resetCFR(stack1,2);

%stack =onesingletone(stack, chan, prof, amp1,     phase,    freqHz)
stack1 = onesingletone( stack1,    0,    0,    1, phasediff, refFreq);
stack1 = onesingletone( stack1,    1,    0,    1,        0, refFreq);

% Phase lock on Rack A
[knownCFR1, stack1] = setClearPhaseAccum(stack1, 2, 1, knownCFR1);
stack1 = flexupdateboth(stack1);
[knownCFR1, stack1]= setClearPhaseAccum(stack1, 2, 0, knownCFR1);
stack1 = waitforRackA(stack1,2);
stack1 = flexupdateboth(stack1);        % start freqs on 1st trigger
                                        % needed for phase-matching
stack1 = flexflush(t1, stack1);

%%  set other channel to be sweeping freq reference
[t2, stack2] =openconn('192.168.0.45', 1);
[knownCFR2, stack2] = resetCFR(stack2,2);

% Set DRG to Ramp Mode
[knownCFR2, stack2] = setDRGFreq(  stack2, 0, knownCFR2);
[knownCFR2, stack2] = setDRGEnable(stack2, 0,  1, knownCFR2);
% ensure Ramp direction is down
stack2 = rampdown(stack2, 0);

% Establish a phase diffrence between channels
%stack =onesingletone(stack, chan, prof, amp1,     phase,    freqHz)
if startFreq < midFreq
    stack2 = onesingletone( stack2,    0,    0,    1,       0, startFreq+refFreq);
else    % need to use mirror freqs, mirror phase too
    stack2 = onesingletoneM(stack2,    0,    0,    1,       0, startFreq+refFreq);
end

% eat one, when A/B phase lock
stack2 = waitforRackA(stack2, 0);
stack2 = flexupdateone(stack2,0);

% create a list of frequencies and sweep times
freqlist = [startFreq+refFreq,midFreq+refFreq,endFreq+refFreq];
timelist = [sweepTime1,sweepTime2];

% put the frequency sweeping info into the stack
stack2 = multifreqtime(stack2, 2, freqlist, timelist, knownCFR2);

% Convert to Words and Send the Ramp parameters
stack2 = flexflush(t2, stack2);


%% report
fclose('all');
disp(datestr(now,'hh:MM:SS'));
clear;

