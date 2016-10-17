#!/usr/bin/env python
import chemcoord as cc
import argparse


def read_input():
    # Calling parser for reading filename
    parser = argparse.ArgumentParser(
        description="""This script takes an filename.xyz file and slices out all
        those rows belonging to points outside a given sphere.
        The sphere is specified with a radius and xyz-coordinates for the center.
        The resulting coordinates are written to filename_changed.xyz.
        Please note that an existing file with this name is overwritten."""
        )
    parser.add_argument("inputfile", type=str,
            help = "Name of the inputfile.")
    parser.add_argument("--outputfile", type=str,
            help = (
                """
                Name of the outputfile. 
                If None is given it takes then name of the inputfile without fileending and appends .zmat
                """
                ),
            default = None
            )
    parser.add_argument(
        "--recursion_level", 
        help="Help text",
        default=2
        )

    args = parser.parse_args()
    return args


args = read_input()

if args.outputfile is None:
    outputfile = ''.join(args.inputfile.split('.')[:-1]) + '.zmat'
else:
    outputfile = args.outputfile

ref_xyz = cc.Cartesian.read_xyz(args.inputfile)
ref_xyz.to_zmat().write(outputfile)
