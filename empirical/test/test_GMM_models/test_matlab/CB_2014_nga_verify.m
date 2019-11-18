#!/usr/bin/env octave

function CB_2014_nga_verify
    % store expected outputs to compare with Python version
    % addpath('.');
    periods = [0.013, 0.024, 0.032, 0.045, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, -1];
    mags = [4.8, 5.5, 6.5, 9.2];
    rrups = [[30, 80]; [29, 30]; [12, 103]; [999, 20]];
    regions = [0, 2, 3, 4];
    ld = [[29, 30, -120]; [0, 45, 25]];
    vs30s = [678, 485, 2504, 145];
    z25 = [999, 32];
    zs = [[999, 17, 17, 18]; [18, 20, 21, 22]; [999, 5, 25, 35]];

    fid = fopen('cb2014.f32','w');

    for p = periods
        for m = mags
            for g = ld
                for r = rrups
                    for f = 0:1
                        for h = zs
                            for v = vs30s
                                for z = z25
                                    for l = regions
                                        [sa, sigma, period_out] = CB_2014_nga(m, p, r(1), r(2), r(3), r(4), h(1), h(2), g(2), g(1), f, v, z, h(3), l)
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
