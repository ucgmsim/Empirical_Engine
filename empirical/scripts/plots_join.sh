#!/usr/bin/env bash

for p in {0..7}; do
    convert "${p}0.png" "${p}1.png" "${p}2.png" +append "${p}a.png"
    convert "${p}3.png" "${p}4.png" "${p}5.png" +append "${p}b.png"
    convert "${p}a.png" "${p}b.png" -append "${p}.png"
done
convert 0.png 1.png 2.png 3.png 4.png 5.png 6.png 7.png "rrup.pdf"
