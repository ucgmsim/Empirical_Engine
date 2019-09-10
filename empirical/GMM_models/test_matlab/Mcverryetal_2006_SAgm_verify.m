#!/usr/bin/env octave

function Mcverryetal_2006_SAgm_verify
    % store expected outputs to compare with Python version
    % addpath('.');
    periods = [-1.0, 0.078, 0.14, 0.18, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3];
    mags = [4.8, 6, 10];
    rrups = [0, 30, 800];
    siteclasses = ['A', 'B', 'C', 'D', 'E'];
    faultstyles = {"normal", "reverse", "oblique", "strikeslip", "interface", "slab"};
    rvols = [82.24, 2.244];
    hcs = [4.422, 39.24];

    fid = fopen('mv2006.f32','w');

    for p = periods
        siteprop.period = p;
        for m = mags
            faultprop.Mw = m;
            for s = siteclasses
                siteprop.siteclass = s;
                for r = rrups
                    siteprop.Rrup = r;
                    for f = 1:length(faultstyles)
                        faultprop.faultstyle = faultstyles{f};
                        for v = rvols
                            siteprop.rvol = v;
                            for h = hcs
                                faultprop.Hc = h;

                                [sa, sigma] = Mcverryetal_2006_SAgm(siteprop, faultprop)
                                fwrite(fid, [sa, sigma], 'float32');
                            end
                        end
                    end
                end
            end
        end
    end

    fclose(fid);
