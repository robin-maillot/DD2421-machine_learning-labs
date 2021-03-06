#
# gccopts.sh   Shell script for configuring MEX-file creation script,
#               mex.
#
# usage:        Do not call this file directly; it is sourced by the
#               mex shell script.  Modify only if you don't like the
#               defaults after running mex.  No spaces are allowed
#               around the '=' in the variable assignment.
#
#               Note: only the gcc side of this script was tested.
#               The FORTRAN variables are lifted directly from
#               mexopts.sh; use that file for compiling FORTRAN
#               MEX-files.
#
# SELECTION_TAGs occur in template option files and are used by MATLAB
# tools, such as mex and mbuild, to determine the purpose of the contents
# of an option file. These tags are only interpreted when preceded by '#'
# and followed by ':'.
#
#SELECTION_TAG_MEX_OPT: Template Options file for building gcc MEXfiles
#
# Copyright (c) 1984-1998 by The MathWorks, Inc.
# All Rights Reserved.
# $Revision: 1.26 $  $Date: 1998/12/16 23:29:08 $
#----------------------------------------------------------------------------
#
    case "$Arch" in
        Undetermined)
#----------------------------------------------------------------------------
# Change this line if you need to specify the location of the MATLAB
# root directory.  The cmex script needs to know where to find utility
# routines so that it can determine the architecture; therefore, this
# assignment needs to be done while the architecture is still
# undetermined.
#----------------------------------------------------------------------------
            MATLAB="$MATLAB"
#
# Determine the location of the GCC libraries
#
	    GCC_LIBDIR=`gcc -v 2>&1 | awk '/.*Reading specs.*/ {print substr($4,0,length($4)-6)}'`
            ;;
        alpha)   # gcc version 2.8.1
#----------------------------------------------------------------------------
            CC='gcc'
            CFLAGS='-mieee'
            CLIBS="-L$GCC_LIBDIR -lgcc"
            COPTIMFLAGS='-O -DNDEBUG'
            CDEBUGFLAGS='-g'
#
            FC='f77'
            FFLAGS='-shared'
            FLIBS='-lUfor -lfor -lFutil'
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD='ld'
            LDFLAGS="-expect_unresolved '*' -shared -hidden -exported_symbol $ENTRYPOINT -exported_symbol mexVersion"
            LDOPTIMFLAGS=''
            LDDEBUGFLAGS=''
#----------------------------------------------------------------------------
            ;;
        hp700)   # gcc version 2.7.2.f.1
# Note: The GNU assembler does not fully support PIC.  From the Info
# documentation:
#
#     The GNU assembler does not fully support PIC.  Currently, you must
#     use some other assembler in order for PIC to work.  We would
#     welcome volunteers to upgrade GAS to handle this; the first part
#     of the job is to figure out what the assembler must do differently.
#
# PIC is necessary for building shared libraries.  Therefore, we need to
# use the HP assembler.
#----------------------------------------------------------------------------
            echo ''
            echo 'Warning: Assembly code generated by gcc may not compile'
            echo 'with the HP assembler'
            echo ''
            CC='gcc'
            CFLAGS='-fPIC -mpa-risc-1-0 -D_HPUX_SOURCE -B/usr/ccs/bin/ -mno-gas'
            CLIBS="-L$GCC_LIBDIR -lgcc"
            COPTIMFLAGS='-O -DNDEBUG'
            CDEBUGFLAGS='-g'
#
            FC='f77'
            FFLAGS='+z'
            FLIBS=''
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD='ld'
            LDFLAGS="-b +e $ENTRYPOINT +e mexVersion"
            LDOPTIMFLAGS=''
            LDDEBUGFLAGS=''
#----------------------------------------------------------------------------
            ;;
        ibm_rs)   # gcc version 2.7.2.f.1
#----------------------------------------------------------------------------
            CC='gcc'
            CFLAGS=''
            CLIBS="-L$MATLAB/bin/$Arch -lmatlbmx -lm"
            COPTIMFLAGS='-O -DNDEBUG'
            CDEBUGFLAGS='-g'
#
            FC='f77'
            FFLAGS=''
            FLIBS="$MATLAB/extern/lib/ibm_rs/fmex1.o -lm"
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD='gcc'
            LDFLAGS="-shared -B/usr/bin/ -Wl,-e$ENTRYPOINT,-bM:SRE,-bI:$MATLAB/extern/lib/ibm_rs/exp.ibm_rs,-bE:$MATLAB/extern/lib/ibm_rs/$MAPFILE"
            LDOPTIMFLAGS='-Wl,-s'
            LDDEBUGFLAGS=''
#----------------------------------------------------------------------------
            ;;
        lnx86)   # gcc version 2.7.2.1
#----------------------------------------------------------------------------
#
# Default to libc5 based development (ie. RedHat4.2)
#
	    CC='gcc'
            if [ -f /etc/redhat-release ]; then
		OS=`cat /etc/redhat-release`
		version=`expr "$OS" : '.*\([0-9][0-9]*\)\.'`
#
# Use this compiler for RedHat5.* systems
#
		if [ "$version" = "5" ]; then
		    CC='i486-linuxlibc5-gcc'
		fi
	    elif [ -f /etc/debian_version ]; then
	        OS=`cat /etc/debian_version`
		version=`expr "$OS" : '.*\([0-9][0-9]*\)\.'`
#
# Use this compiler for Debian 2.* systems
#
		if [ "$version" = "2" ]; then
		    CC='i486-linuxlibc1-gcc'
		fi
	    fi
            CFLAGS=''
            CLIBS=''
            COPTIMFLAGS='-O -DNDEBUG'
            CDEBUGFLAGS='-g'
#
# These flags use f2c and gcc for building FORTRAN MEX-Files
# The fort77 script invokes the f2c command transparently,
# so it can be used like a real FORTRAN compiler.
#
            FC='fort77'
            FFLAGS=''
            FLIBS='-lf2c -Wl,--defsym,MAIN__=mexfunction_'
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD=$CC
            LDFLAGS='-shared'
            LDOPTIMFLAGS=''
            LDDEBUGFLAGS=''
#----------------------------------------------------------------------------
            ;;
        sgi)   # gcc version 2.7.2.2
#----------------------------------------------------------------------------
            CC='gcc'
            CFLAGS=''
            CLIBS="-L$GCC_LIBDIR -lgcc"
            COPTIMFLAGS='-O -DNDEBUG'
            CDEBUGFLAGS='-g'
#
            FC='f77'
            FFLAGS=''
            FLIBS=''
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD="$GCC_LIBDIR/ld"
#            LD="ld"     # Use this line for compiling 32-bit MEX on SGI64
            LDFLAGS="-shared -exported_symbol $ENTRYPOINT -exported_symbol mexVersion"
            LDOPTIMFLAGS=''
            LDDEBUGFLAGS=''
            ;;
#----------------------------------------------------------------------------
        sgi64)   # gcc version 2.8.1
#----------------------------------------------------------------------------
            CC='gcc'
            CFLAGS='-mabi=64'
            CLIBS="-L$GCC_LIBDIR/mabi=64 -lgcc"
            COPTIMFLAGS='-O -DNDEBUG'
            CDEBUGFLAGS='-g'
#
            FC='f77'
            FFLAGS=''
            FLIBS=''
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD='ld'
            LDFLAGS="-64 -shared -exported_symbol $ENTRYPOINT -exported_symbol mexVersion"
            LDOPTIMFLAGS=''
            LDDEBUGFLAGS=''
            ;;
#----------------------------------------------------------------------------
        sol2)   # gcc version 2.7.2.f.1
#----------------------------------------------------------------------------
            CC='gcc'
            CFLAGS='-fPIC'
            CLIBS="-L$GCC_LIBDIR -lgcc"
            COPTIMFLAGS='-O -DNDEBUG'
            CDEBUGFLAGS='-g'
#
            FC='f77'
            FFLAGS='-G'
            FLIBS=''
            FOPTIMFLAGS='-O'
            FDEBUGFLAGS='-g'
#
            LD='/usr/ccs/bin/ld'
            LDFLAGS="-G -M $MATLAB/extern/lib/sol2/$MAPFILE"
            LDOPTIMFLAGS=''
            LDDEBUGFLAGS=''
#----------------------------------------------------------------------------
            ;;
    esac
#############################################################################
#
# Architecture independent lines:
#
#     Set and uncomment any lines which will apply to all architectures.
#
#----------------------------------------------------------------------------
#           CC="$CC"
#           CFLAGS="$CFLAGS"
#           COPTIMFLAGS="$COPTIMFLAGS"
#           CDEBUGFLAGS="$CDEBUGFLAGS"
#           CLIBS="$CLIBS"
#
#           FC="$FC"
#           FFLAGS="$FFLAGS"
#           FOPTIMFLAGS="$FOPTIMFLAGS"
#           FDEBUGFLAGS="$FDEBUGFLAGS"
#           FLIBS="$FLIBS"
#
#           LD="$LD"
#           LDFLAGS="$LDFLAGS"
#           LDOPTIMFLAGS="$LDOPTIMFLAGS"
#           LDDEBUGFLAGS="$LDDEBUGFLAGS"
#----------------------------------------------------------------------------
#############################################################################
#EOF--------------------------------------------------------------------
# diff gccopts.sh(new) gccopts.sh(old)
# date: Fri Mar 26 09:54:29 MET 1999
# 24c24
# < # $Revision: 1.26 $  $Date: 1998/12/16 23:29:08 $
# ---
# > # $Revision: 1.13 $  $Date: 1997/12/05 20:18:32 $
# 42c42
# <         alpha)   # gcc version 2.8.1
# ---
# >         alpha)   # gcc version 2.7.2
# 43a44,46
# >             echo ''
# >             echo 'Warning: MEX-Files built on alpha using gcc are not IEEE compliant'
# >             echo ''
# 45c48
# <             CFLAGS='-mieee'
# ---
# >             CFLAGS='-ansi'
# 62c65
# <         hp700)   # gcc version 2.7.2.f.1
# ---
# >         hp700)   # gcc version 2.7.2
# 79c82
# <             CFLAGS='-fPIC -mpa-risc-1-0 -D_HPUX_SOURCE -B/usr/ccs/bin/ -mno-gas'
# ---
# >             CFLAGS='-ansi -fPIC -mpa-risc-1-0 -D_HPUX_SOURCE -B/usr/ccs/bin/ -mno-gas'
# 96c99
# <         ibm_rs)   # gcc version 2.7.2.f.1
# ---
# >         ibm_rs)   # gcc version 2.7.2
# 99,100c102,103
# <             CFLAGS=''
# <             CLIBS="-L$MATLAB/bin/$Arch -lmatlbmx -lm"
# ---
# >             CFLAGS='-ansi'
# >             CLIBS='-lm'
# 116c119
# <         lnx86)   # gcc version 2.7.2.1
# ---
# >         lnx86)   # gcc version 2.7.2
# 118,141c121,122
# < #
# < # Default to libc5 based development (ie. RedHat4.2)
# < #
# < 	    CC='gcc'
# <             if [ -f /etc/redhat-release ]; then
# < 		OS=`cat /etc/redhat-release`
# < 		version=`expr "$OS" : '.*\([0-9][0-9]*\)\.'`
# < #
# < # Use this compiler for RedHat5.* systems
# < #
# < 		if [ "$version" = "5" ]; then
# < 		    CC='i486-linuxlibc5-gcc'
# < 		fi
# < 	    elif [ -f /etc/debian_version ]; then
# < 	        OS=`cat /etc/debian_version`
# < 		version=`expr "$OS" : '.*\([0-9][0-9]*\)\.'`
# < #
# < # Use this compiler for Debian 2.* systems
# < #
# < 		if [ "$version" = "2" ]; then
# < 		    CC='i486-linuxlibc1-gcc'
# < 		fi
# < 	    fi
# <             CFLAGS=''
# ---
# >             CC='gcc'
# >             CFLAGS='-ansi'
# 146,150c127
# < # These flags use f2c and gcc for building FORTRAN MEX-Files
# < # The fort77 script invokes the f2c command transparently,
# < # so it can be used like a real FORTRAN compiler.
# < #
# <             FC='fort77'
# ---
# >             FC='f77'
# 152c129
# <             FLIBS='-lf2c -Wl,--defsym,MAIN__=mexfunction_'
# ---
# >             FLIBS=''
# 156,157c133,134
# <             LD=$CC
# <             LDFLAGS='-shared'
# ---
# >             LD='gcc'
# >             LDFLAGS='-shared -rdynamic'
# 162c139
# <         sgi)   # gcc version 2.7.2.2
# ---
# >         sgi)   # gcc version 2.6.0
# 177,178c154
# < #            LD="ld"     # Use this line for compiling 32-bit MEX on SGI64
# <             LDFLAGS="-shared -exported_symbol $ENTRYPOINT -exported_symbol mexVersion"
# ---
# >             LDFLAGS="-shared -U -Bsymbolic -exported_symbol $ENTRYPOINT -exported_symbol mexVersion"
# 183c159,161
# <         sgi64)   # gcc version 2.8.1
# ---
# >         sgi64)   # gcc version 2.6.0
# > # R8000 only: The default action of mex is to generate full MIPS IV
# > #             (R8000) instruction set.
# 184a163,168
# >             echo ''
# >             echo 'MEX-Files built with gcc are not supported on sgi64'
# >             echo ''
# >             cleanup
# > 	    exit 1
# > #
# 186,187c170,171
# <             CFLAGS='-mabi=64'
# <             CLIBS="-L$GCC_LIBDIR/mabi=64 -lgcc"
# ---
# >             CFLAGS=''
# >             CLIBS="-L$GCC_LIBDIR -lgcc"
# 198c182
# <             LDFLAGS="-64 -shared -exported_symbol $ENTRYPOINT -exported_symbol mexVersion"
# ---
# >             LDFLAGS="-mips4 -64 -shared -U -Bsymbolic -exported_symbol $ENTRYPOINT -exported_symbol mexVersion"
# 203c187
# <         sol2)   # gcc version 2.7.2.f.1
# ---
# >         sol2)   # gcc version 2.6.3
# 206c190
# <             CFLAGS='-fPIC'
# ---
# >             CFLAGS='-ansi -fPIC'
# 221a206,229
# >             ;;
# >         sun4)   # gcc version 2.6.3
# > #----------------------------------------------------------------------------
# > # A dry run of the appropriate compiler is done in the mex script to
# > # generate the correct library list. Use -v option to see what
# > # libraries are actually being linked in.
# > #----------------------------------------------------------------------------
# >             CC='gcc'
# >             CFLAGS='-ansi -Dsparc -DMEXSUN4'
# >             CLIBS="$MATLAB/extern/lib/sun4/libmex.a -lm"
# >             COPTIMFLAGS='-O -DNDEBUG'
# >             CDEBUGFLAGS='-g'
# > #
# >             FC='f77'
# >             FFLAGS=''
# >             FLIBS="$MATLAB/extern/lib/sun4/libmex.a -lm"
# >             FOPTIMFLAGS='-O'
# >             FDEBUGFLAGS='-g'
# > #
# >             LD='ld'
# >             LDFLAGS='-d -r -u _mex_entry_pt -u _mexFunction'
# >             LDOPTIMFLAGS='-x'
# >             LDDEBUGFLAGS=''
# > #----------------------------------------------------------------------------
