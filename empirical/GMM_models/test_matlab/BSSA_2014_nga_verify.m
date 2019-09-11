#!/usr/bin/env octave

function BSSA_2014_nga_verify
    % store expected outputs to compare with Python version
    % addpath('.');
    periods = [-1, 0.011, 0.02, 0.035, 0.06, 0.066, 0.9, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10];
    mags = [4.4, 5, 5.5, 9.2];
    rrups = [0, 98, 2345];
    regions = [0, 1, 2, 3, 4];
    vs30s = [134, 760, 1357];
    z1 = [999, 3, 243];

    fid = fopen('bssa2014.f32','w');

    for p = periods
        for m = mags
            for r = rrups
                for t = 0:3
                    for v = vs30s
                        for z = z1
                            for l = regions
                                [sa, sigma, period_out] = BSSA_2014_nga(m, p, r, t, l, z, v)
                                fwrite(fid, [sa, sigma], 'float32');
                            end
                        end
                    end
                end
            end
        end
    end

    fclose(fid);
