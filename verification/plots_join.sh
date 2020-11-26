#!/usr/bin/env bash

for s in r m; do 
    for t in {0..1}; do
        for p in {0..7}; do
            convert "${s}${t}${p}0.png" "${s}${t}${p}1.png" "${s}${t}${p}2.png" +append "${s}${t}${p}a.png"
            convert "${s}${t}${p}3.png" "${s}${t}${p}4.png" "${s}${t}${p}5.png" +append "${s}${t}${p}b.png"
            convert "${s}${t}${p}a.png" "${s}${t}${p}b.png" -append "${s}${t}${p}.png"
        done
        convert ${s}${t}{0..7}.png "${s}${t}.pdf"
    done
done

for t in {0..1}; do
    convert "s${t}00.png" "s${t}01.png" "s${t}02.png" +append "s${t}0.png"
    convert "s${t}10.png" "s${t}11.png" "s${t}12.png" +append "s${t}1.png"
    convert "s${t}0.png" "s${t}1.png" -append "s${t}.png"
    convert s${t}.png "s${t}.pdf"
done
