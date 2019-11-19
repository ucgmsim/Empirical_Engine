#!/usr/bin/env octave

function ASK_2014_nga_verify
    % store expected outputs to compare with Python version
    % addpath('.');
    periods = [0.012, 0.018, 0.03, 0.05, 0.07, 0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6.5, 8, 10, -1];
    mags = [4.8, 5, 6.5, 9.2];
    rrups = [[30, 80]; [29, 30]; [12, 103]; [999, 20]; [8, 18]; [23, 68]];
    regions = [0, 2, 3, 6];
    ld = [[29, 30, -120]; [0, 45, 25]];
    vs30s = [[67, 485, 2504, 185]; [0, 0, 1, 1]];
    z10 = [999, 3, 243];

    fid = fopen('ask2014.f32','w');

    for p = periods
        for m = mags
            for g = ld
                for r = rrups
                    for f = 0:1
                        for h = 0:1
                            for v = vs30s
                                for z = z10
                                    for l = regions
                                        [sa, sigma, period_out] = ASK_2014_nga(m, p, r(1), r(2), r(3), r(4), r(5), g(2), g(1), f, h, r(6), z, v(1), v(2), l);
                                        fwrite(fid, [sa, sigma], 'float32');
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    fclose(fid);
