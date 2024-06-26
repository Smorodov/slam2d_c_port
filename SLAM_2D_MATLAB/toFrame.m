function [pf, PF_f, PF_p] = toFrame(F , p)
% TOFRAME transform point P from global frame to frame F
%
% In:
% F : reference frame F = [f x ; f y ; f alpha]
% p : point in global frame p = [p x ; p y]
% Out:
% pf: point in frame F
% PF f: Jacobian wrt F
% PF p: Jacobian wrt p
% (c) 2010, 2011, 2012 Joan Sola
t = F(1:2);
a = F(3);
R = [cos(a) -sin(a) ; sin(a) cos(a)];
pf = R' * (p - t);
if nargout > 1 % Jacobians requested
px = p(1);
py = p(2);
x = t(1);
y = t(2);
PF_f = [...
[ -cos(a), -sin(a), cos(a)*(py - y) - sin(a)*(px - x)]
[ sin(a), -cos(a), -cos(a)*(px - x) - sin(a)*(py - y)]];
PF_p = R';
end
end