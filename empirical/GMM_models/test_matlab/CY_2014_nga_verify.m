#!/usr/bin/env octave

function CY_2014_nga_verify
    % store expected outputs to compare with Python version
    % addpath('.');
    periods = [-1, 0.011, 0.022, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.6, 2.4, 2.8, 4.3, 6, 8, 9];
    mags = [4.2, 6.2, 7.2];
    rrups = [[12.6, 1552.22]; [10, 1000]; [3, 24]; [2, 20]];
    ld = [[30, 150, 20, -120, -60]; [45, 60, 80, 110, 10]];
    vs30s = [[67, 485, 2504, 185]; [0, 0, 1, 1]];
    z10 = [999, 3, 243];

    fid = fopen('cy2014.f32','w');

    for p = periods
        for m = mags
            for g = ld
                for r = rrups
                    for f = 0:1
                        for v = vs30s
                            for z = z10
                                for l = 0:5
                                    [sa, sigma, period_out] = CY_2014_nga(m, p, r(1), r(2), r(3), r(4), g(2), g(1), z, v(1), f, v(2), l)
                                    fwrite(fid, [sa, sigma], 'float32');
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    fclose(fid);
