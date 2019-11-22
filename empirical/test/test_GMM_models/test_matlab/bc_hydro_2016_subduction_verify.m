#!/usr/bin/env octave

function bc_hydro_2016_subduction_verify
    % store expected outputs to compare with Python version
    % addpath('.');
    periods = [0.021, 0.06, 0.075, 0.099, 0.14, 0.201, 0.262, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 10];
    % all periods, <> 7.8.
    mags = [7.5, 7.8, 8.0, 8.2];
    % all periods, backarc sites? <> 85 (intraslab) <> 100 (interface)
    rrups = [0, 12.6, 90, 1052.22];
    % all periods, <> 1000 and by Vlin constant
    vs30s = [67, 485, 2504];
    % all periods, <> 120 [intraslab]
    hyps = [131, 0, NaN];

    fid = fopen('bchydro.f32','w');

    for p = periods
        for m = mags
            for h = hyps
                for r = rrups
                    for f = 0:1
                        for v = vs30s
                            [sa, sigma] = bc_hydro_2016_subduction(p, m, r, f, v, isnan(h), h);
                            fwrite(fid, [sa, sigma], 'float32');
                        end
                    end
                end
            end
        end
    end

    fclose(fid);
