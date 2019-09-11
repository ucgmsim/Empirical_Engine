function [SA,sigma_SA]=Mcverryetal_2006_SAgm(siteprop,faultprop)

%puropse: provide the geometric mean and dispersion in the McVerry attenuation
%relationship for a given M,R and faulting style, soil conditions etc

%reference: McVerry GH, Zhao JX, Abrahamson NA, Somerville PG. New Zealand
%Accelerations Response Spectrum Attenuation Relations for Crustal and
%Subduction Zone Earthquakes.  Bulletin of the New Zealand Society of
%Earthquake Engineering. Vol 39, No 4. March 2006

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Inputvariables
%siteprop
%   Rrup -- Shortest distance from source to site (km) (i.e. Rrup)
%   T   -- period of vibration to compute attenuation for
%          uses linear interpolation between actual parameter values
%   siteclass - 'A','B','C','D','E'; as per NZS1170.5
%   rvol-- length in km of the part of the source to site distance in volcanic
%          zone (not needed for slab event)

%faultprop
%   Mw   -- Moment magnitude
%   faultstyle   
%       - crustal events - 'normal','reverse','oblique'
%       - subduction events - 'interface','slab'
%                           Hc  -- the centroid depth in km

%Output Variables:
% SA           = median SA  (or PGA or PGV) (geometric mean)
% sigma_SA     = lognormal standard deviation in SA
                 %sigma_SA(1) = total std
                 %sigma_SA(2) = interevent std
                 %sigma_SA(3) = intraevent std

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%
%Parameters 
%%%%%%%%%%%%%%%%
%first column corresponds to the 'prime' values
period=[    -1.0,      0.0,    0.075,      0.1,      0.2,      0.3,      0.4,      0.5,     0.75,      1.0,      1.5,      2.0,      3.0];
C1=    [ 0.14274,  0.07713,  1.22050,  1.53365,  1.22565,  0.21124, -0.10541, -0.14260, -0.65968, -0.51404, -0.95399, -1.24167, -1.56570]; 
C3AS=  [     0.0,      0.0,     0.03,    0.028,  -0.0138,   -0.036,  -0.0518,  -0.0635,  -0.0862,   -0.102,    -0.12,    -0.12, -0.17260];
C4AS=   -0.144;
C5=    [-0.00989, -0.00898, -0.00914, -0.00903, -0.00975, -0.01032, -0.00941, -0.00878, -0.00802, -0.00647, -0.00713, -0.00713, -0.00623];
C6AS=   0.17;
C8=    [-0.68744, -0.73728, -0.93059, -0.96506, -0.75855, -0.52400, -0.50802, -0.52214, -0.47264, -0.58672, -0.49268, -0.49268, -0.52257];
C10AS= [     5.6,      5.6,     5.58,      5.5,      5.1,      4.8,     4.52,      4.3,      3.9,      3.7,     3.55,     3.55,      3.5];
C11=   [ 8.57343,  8.08611,  8.69303,  9.30400, 10.41628,  9.21783,   8.0115,  7.87495,  7.26785,  6.98741,  6.77543,  6.48775,  5.05424];
C12y=   1.414;
C13y=  [     0.0,      0.0,      0.0,  -0.0011,  -0.0027,  -0.0036,  -0.0043,  -0.0048,  -0.0057,  -0.0064,  -0.0073,  -0.0073,  -0.0089];
C15=   [  -2.552,   -2.552,   -2.707,   -2.655,   -2.528,   -2.454,   -2.401,    -2.36,   -2.286,   -2.234,    -2.16,    -2.16,   -2.033]; 
C17=   [-2.56592, -2.49894, -2.55903, -2.61372, -2.70038, -2.47356, -2.30457, -2.31991, -2.28460, -2.28256, -2.27895, -2.27895, -2.05560];
C18y=   1.7818;
C19y=   0.554;
C20=   [ 0.01545,   0.0159,  0.01821,  0.01737,  0.01531,  0.01304,  0.01426,  0.01277,  0.01055,  0.00927,  0.00748,  0.00748, -0.00273];
C24=   [-0.49963, -0.43223, -0.52504, -0.61452, -0.65966, -0.56604, -0.33169, -0.24374, -0.01583,  0.02009, -0.07051, -0.07051, -0.23967];
C29=   [ 0.27315,  0.38730,  0.27879,  0.28619,  0.34064,  0.53213,  0.63272,  0.58809,  0.50708,  0.33002,  0.07445,  0.07445,  0.09869]; 
C30AS= [   -0.23,    -0.23,    -0.28,    -0.28,   -0.245,   -0.195,    -0.16,   -0.121,    -0.05,      0.0,     0.04,     0.04,     0.04];
C32=    0.2;
C33AS= [    0.26,     0.26,     0.26,     0.26,     0.26,    0.198,    0.154,    0.119,    0.057,    0.013,   -0.049,   -0.049,   -0.156];
C43=   [-0.33716, -0.31036, -0.49068, -0.46604, -0.31282, -0.07565,  0.17615,  0.34775,  0.72380,  0.89239,  0.77743,  0.77743,  0.60938];
C46=   [-0.03255,  -0.0325, -0.03441, -0.03594, -0.03823, -0.03535, -0.03354, -0.03211, -0.02857,   -0.025, -0.02008, -0.02008, -0.01587];
Sigma6=[  0.4871,   0.5099,   0.5297,   0.5401,   0.5599,   0.5456,   0.5556,   0.5658,   0.5611,   0.5573,   0.5419,   0.5419,   0.5809]; 
Sigslope=[-0.1011,  -0.0259,  -0.0703,  -0.0292,   0.0172,  -0.0566,  -0.1064,  -0.1123,  -0.0836,  -0.0620,   0.0385,   0.0385,   0.1403];
Tau=   [  0.2677,   0.2469,   0.3139,   0.3017,   0.2583,   0.1967,   0.1802,   0.1440,   0.1871,   0.2073,   0.2405,   0.2405,   0.2053];
  
M = faultprop.Mw;
R=siteprop.Rrup;

T=siteprop.period;
% interpolate between periods if neccesary
if (length(find(abs((period-T))<0.0001))==0)
    T_low=max(period(find(period<T)));
    T_hi=min(period(find(period>T)));
    
    siteprop.period=T_low;
    [SA_low,sigma_SA_low]=Mcverryetal_2006_SAgm(siteprop,faultprop);
    siteprop.period=T_hi;
    [SA_high,sigma_SA_high]=Mcverryetal_2006_SAgm(siteprop,faultprop);
    siteprop.period=T;
    
    if T_low>eps %log interpolation
        x=[log(T_low) log(T_hi)];
        Y_sa=[log(SA_low) log(SA_high)];
        SA_sigma=[sigma_SA_low' sigma_SA_high'];
        SA=exp(interp1(x,Y_sa,log(T)));
        for i=1:3
            sigma_SA(i) = interp1(x,SA_sigma(i,:),log(T));
        end
    else    %inear interpolation
        x=[T_low T_hi];
        Y_sa=[SA_low SA_high];
        SA_sigma=[sigma_SA_low' sigma_SA_high'];
        SA=interp1(x,Y_sa,T);
        for i=1:3
            sigma_SA(i) = interp1(x,SA_sigma(i,:),T);
        end
    end
    
else
    %error if greater than max period
    if T>period(length(period))
        fprintf('Error: Mcverryetal_2006_SAgm.m: The vibration period %5.2f is greater than the maximum T=3.0 s \n',T);
        return;
    end
%   otherwise
    i = find(abs((period - T)) < 0.0001); % Identify the period 
    
    %site class
    delC=0; delD=0;
    if length(regexp(siteprop.siteclass,'A','match'))~=0  %A
    elseif length(regexp(siteprop.siteclass,'B','match'))~=0  %B
    elseif length(regexp(siteprop.siteclass,'C','match'))~=0  %C
        delC=1;
    elseif length(regexp(siteprop.siteclass,'D','match'))~=0  %D
        delD=1;
    elseif length(regexp(siteprop.siteclass,'E','match'))~=0  %E
    else
        fprintf('Mcverryetal_2006_SAgm.m: The site class %s is not supported \n',cell2mat(siteprop.siteclass));
    end    
        
    if length(regexp(faultprop.faultstyle,'normal','match'))~=0  %normal faulting
        rvol=siteprop.rvol; %volcanic path term
        CN=-1;        CR=0;
        CS=0; %crustal event
    elseif length(regexp(faultprop.faultstyle,'reverse','match'))~=0  %reverse faulting
        rvol=siteprop.rvol; %volcanic path term
        CN=0;        CR=1;
        CS=0; %crustal event
    elseif length(regexp(faultprop.faultstyle,'oblique','match'))~=0  %reverse/oblique faulting
        rvol=siteprop.rvol; %volcanic path term
        CN=0;        CR=0.5;
        CS=0; %crustal event
    elseif length(regexp(faultprop.faultstyle,'strikeslip','match'))~=0  %strikeslip
        rvol=siteprop.rvol; %volcanic path term
        CN=0;        CR=0;
        CS=0; %crustal event
    elseif length(regexp(faultprop.faultstyle,'slab','match'))~=0  %subduction slab
        rvol=0; %volcanic path term - a value of zero used as it doesnt matter in equation for slab event
        SI=0;        DS=1;
        CS=1; %subduction event
    elseif length(regexp(faultprop.faultstyle,'interface','match'))~=0  %subduction interface
        rvol=siteprop.rvol; %volcanic path term
        SI=1;        DS=0;
        CS=1; %subduction event
    else
        fprintf('Error Mcverryetal_2006_SAgm.m: the fauly type %s is not avaliable \n',cell2mat(faultprop.faultstyle));
    end
        
    if CS==0 %crustal prediction equation
        PGA_AB=     exp(C1(1)+C4AS*(M-6)+C3AS(1)*(8.5-M)^2+C5(1)*R+(C8(1)+C6AS*(M-6))*log(sqrt(R^2+C10AS(1)^2))+C46(1)*rvol+C32*CN+C33AS(1)*CR);
        PGA_primeAB=exp(C1(2)+C4AS*(M-6)+C3AS(2)*(8.5-M)^2+C5(2)*R+(C8(2)+C6AS*(M-6))*log(sqrt(R^2+C10AS(2)^2))+C46(2)*rvol+C32*CN+C33AS(2)*CR);
        Sa_primeAB= exp(C1(i)+C4AS*(M-6)+C3AS(i)*(8.5-M)^2+C5(i)*R+(C8(i)+C6AS*(M-6))*log(sqrt(R^2+C10AS(i)^2))+C46(i)*rvol+C32*CN+C33AS(i)*CR);
        
    elseif CS==1 %subduction attenuation
        Hc=faultprop.Hc;    %centroid depth (similar, but different to focal depth?)
        
        PGA_AB=     exp(C11(1)+(C12y+(C15(1)-C17(1))*C19y)*(M-6)+C13y(1)*(10-M)^3+C17(1)*log(R+C18y*exp(C19y*M))+C20(1)*Hc+C24(1)*SI+C46(1)*rvol*(1-DS));
        PGA_primeAB=exp(C11(2)+(C12y+(C15(2)-C17(2))*C19y)*(M-6)+C13y(2)*(10-M)^3+C17(2)*log(R+C18y*exp(C19y*M))+C20(2)*Hc+C24(2)*SI+C46(2)*rvol*(1-DS));
        Sa_primeAB= exp(C11(i)+(C12y+(C15(i)-C17(i))*C19y)*(M-6)+C13y(i)*(10-M)^3+C17(i)*log(R+C18y*exp(C19y*M))+C20(i)*Hc+C24(i)*SI+C46(i)*rvol*(1-DS));
    end
    
    PGA_CD=          PGA_AB*exp(C29(1)*delC+(C30AS(1)*log(PGA_primeAB+0.03)+C43(1))*delD);
    PGA_primeCD=PGA_primeAB*exp(C29(2)*delC+(C30AS(2)*log(PGA_primeAB+0.03)+C43(2))*delD);
    Sa_primeCD=  Sa_primeAB*exp(C29(i)*delC+(C30AS(i)*log(PGA_primeAB+0.03)+C43(i))*delD);
    Sa=Sa_primeCD*PGA_CD/PGA_primeCD;
    
    %standard deviation
    if M<5
        sig_intra=Sigma6(i)-Sigslope(i);
    elseif M>7
        sig_intra=Sigma6(i)+Sigslope(i);
    else
        sig_intra=Sigma6(i)+Sigslope(i)*(M-6);
    end
    
    %outputting
    SA=Sa;
    sigma_SA(1)=sqrt(sig_intra^2+Tau(i)^2);
    sigma_SA(2)=sig_intra;
    sigma_SA(3)=Tau(i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
