function maximize(fig)
% MAXIMIZE Size a window to fill the entire screen.
%
% maximize(HANDLE fig)
%  Will size the window with handle fig such that it fills the entire screen.
% Original author: Bill Finger, Creare Inc.
% Free for redistribution as long as credit comments are preserved.
%

if nargin==0, fig=gcf; end

units=get(fig,'units');
set(fig,'units','normalized','outerposition',[0 0 1 1]);
set(fig,'units',units);