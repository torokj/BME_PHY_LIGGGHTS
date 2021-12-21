/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║  ███╗██║  ███╗██║  ███╗███████║   ██║   ███████╗
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:
    This file is from LAMMPS, but has been modified. Copyright for
    modification:

    Copyright 2012-     DCS Computing GmbH, Linz
    Copyright 2009-2012 JKU Linz

    Copyright of original file:
    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
    http://lammps.sandia.gov, Sandia National Laboratories
    Steve Plimpton, sjplimp@sandia.gov

    Copyright (2003) Sandia Corporation.  Under the terms of Contract
    DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
    certain rights in this software.  This software is distributed under
    the GNU General Public License.
------------------------------------------------------------------------- */

#include <cmath>
#include <stdlib.h>
#include <string.h>
#include "dump_mozi.h"
#include "atom.h"
#include "force.h"
#include "domain.h"
#include "region.h"
#include "group.h"
#include "input.h"
#include "variable.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "fix.h"
#include "memory.h"
#include "error.h"
#include "compute_pair_gran_local.h"
#include "fix_wall_gran.h"

using namespace LAMMPS_NS;

// customize by adding keyword
// also customize compute_atom_property.cpp

enum{ID,MOL,TYPE,ELEMENT,MASS,
     X,Y,Z,XS,YS,ZS,XSTRI,YSTRI,ZSTRI,XU,YU,ZU,XUTRI,YUTRI,ZUTRI,
     XSU,YSU,ZSU,XSUTRI,YSUTRI,ZSUTRI,
     IX,IY,IZ,
     VX,VY,VZ,FX,FY,FZ,
     Q, MUX,MUY,MUZ,MU,RADIUS,DIAMETER,
     OMEGAX,OMEGAY,OMEGAZ,ANGMOMX,ANGMOMY,ANGMOMZ,
     TQX,TQY,TQZ,SPIN,ERADIUS,ERVEL,ERFORCE,
     COMPUTE,FIX,VARIABLE,
     DENSITY, RHO, P,
     SHAPEX, SHAPEY, SHAPEZ,
     QUAT1, QUAT2, QUAT3, QUAT4,
     BLOCKINESS1, BLOCKINESS2,
     INERTIAX, INERTIAY, INERTIAZ}; 
enum{LT,LE,GT,GE,EQ,NEQ};
enum{INT,DOUBLE,STRING,ORIENT};

enum{Pos,Orient,Coord,Velo,Force,AngVelo,Torque,TrVelo,TrForce,TrAngVelo};
const char *parnam[]={ "Pos","Orient","Coord","Velo","Force",
  "AngVelo","Torque","TrVelo","TrForce","TrAngVelo","" };
int parlength[]={3,4,1,3,3,3,3,3,3,3};

#define ONEFIELD 32
#define DELTA 1048576
#define SIGN(a) ( (a>0.0)? 1.0 : ( (a<0.0) ? -1.0 : 0.0) )

/* ---------------------------------------------------------------------- */

DumpMozi::DumpMozi(LAMMPS *lmp, int narg, char **arg) :
  Dump(lmp, narg, arg)
{
  if (narg == 5) error->all(FLERR,"No dump mozi arguments specified");

  clearstep = 1;

  nevery = force->inumeric(FLERR,arg[3]);

  // size_one may be shrunk below if additional optional args exist

/*  size_one = nfield = narg - 5;*/
  // process attributes

  haveforces = 0;
  size_one = nfield = pre_parse_fields(narg,arg);

  pack_choice = new FnPtrPack[nfield];
  vtype = new int[nfield];
  stl = new char*[stl_num];
  wall = new class ComputePairGranLocal*[stl_num];

  buffer_allow = 1;
  buffer_flag = 1;
  iregion = -1;
  idregion = NULL;
  nthresh = 0;
  thresh_array = NULL;
  thresh_op = NULL;
  thresh_value = NULL;

  // computes, fixes, variables which the dump accesses

  memory->create(field2index,nfield,"dump:field2index");
  memory->create(argindex,nfield,"dump:argindex");

  parse_fields(narg,arg);

  ncompute = 0;
  id_compute = NULL;
  compute = NULL;

  nfix = 0;
  id_fix = NULL;
  fix = NULL;

  nvariable = 0;
  id_variable = NULL;
  variable = NULL;
  vbuf = NULL;

  // atom selection arrays

  maxlocal = 0;
  choose = NULL;
  dchoose = NULL;
  clist = NULL;

  // element names

  ntypes = atom->ntypes;
  typenames = NULL;

  // setup format strings

  vformat = new char*[size_one];

  format_default = new char[3*size_one+1];
  format_default[0] = '\0';

  for (int i = 0; i < size_one; i++) {
    if (vtype[i] == INT) strcat(format_default,"%d ");
    else if (vtype[i] == DOUBLE) strcat(format_default,"%g ");
    else if (vtype[i] == STRING) strcat(format_default,"%s ");
    vformat[i] = NULL;
  }

  // setup column string

  int n = 0;
  for (int iarg = 5; iarg < narg; iarg++) n += strlen(arg[iarg]) + 2;
  columns = new char[n];
  columns[0] = '\0';
  for (int iarg = 5; iarg < narg; iarg++) {
    strcat(columns,arg[iarg]);
    strcat(columns," ");
  }

  label = NULL; 
  I = sizeof(int);
  D = sizeof(double);
  binary = 1;
}

/* ---------------------------------------------------------------------- */

DumpMozi::~DumpMozi()
{
  delete [] pack_choice;
  delete [] vtype;
  memory->destroy(field2index);
  memory->destroy(argindex);

  delete [] idregion;
  memory->destroy(thresh_array);
  memory->destroy(thresh_op);
  memory->destroy(thresh_value);

  for (int i = 0; i < ncompute; i++) delete [] id_compute[i];
  memory->sfree(id_compute);
  delete [] compute;

  for (int i = 0; i < nfix; i++) delete [] id_fix[i];
  memory->sfree(id_fix);
  delete [] fix;

  for (int i = 0; i < nvariable; i++) delete [] id_variable[i];
  memory->sfree(id_variable);
  delete [] variable;
  for (int i = 0; i < nvariable; i++) memory->destroy(vbuf[i]);
  delete [] vbuf;

  memory->destroy(choose);
  memory->destroy(dchoose);
  memory->destroy(clist);

  if (typenames) {
    for (int i = 1; i <= ntypes; i++) delete [] typenames[i];
    delete [] typenames;
  }

  for (int i = 0; i < size_one; i++) delete [] vformat[i];
  delete [] vformat;
  delete [] label; 
  delete [] columns;
}

/* ---------------------------------------------------------------------- */

void DumpMozi::init_style()
{
  delete [] format;
  char *str;
  if (format_user) str = format_user;
  else str = format_default;

  int n = strlen(str) + 1;
  format = new char[n];
  strcpy(format,str);

  // default for element names = C

  if (typenames == NULL) {
    typenames = new char*[ntypes+1];
    for (int itype = 1; itype <= ntypes; itype++) {
      typenames[itype] = new char[2];
      strcpy(typenames[itype],"C");
    }
  }

  // tokenize the format string and add space at end of each format element

/*  char *ptr;
  printf("XXX %d\n",size_one);
  for (int i = 0; i < size_one; i++) {
    printf("III %d\n",i);
    if (i == 0) ptr = strtok(format," \0");
    else ptr = strtok(NULL," \0");
    if (ptr == NULL) error->all(FLERR,"Dump_modify format string is too short");
    delete [] vformat[i];
    vformat[i] = new char[strlen(ptr) + 2];
    strcpy(vformat[i],ptr);
    vformat[i] = strcat(vformat[i]," ");
  }*/

  // setup boundary string

  domain->boundary_string(boundstr);

  if (binary) write_choice = &DumpMozi::write_binary;
  else if (buffer_flag == 1) write_choice = &DumpMozi::write_string;
  else write_choice = &DumpMozi::write_lines;

  // find current ptr for each compute,fix,variable
  // check that fix frequency is acceptable

  int icompute;
  for (int i = 0; i < ncompute; i++) {
    icompute = modify->find_compute(id_compute[i]);
    if (icompute < 0) error->all(FLERR,"Could not find dump mozi compute ID");
    compute[i] = modify->compute[icompute];
  }

  int ifix;
  for (int i = 0; i < nfix; i++) {
    ifix = modify->find_fix(id_fix[i]);
    if (ifix < 0) error->all(FLERR,"Could not find dump mozi fix ID");
    fix[i] = modify->fix[ifix];
    if (nevery % modify->fix[ifix]->peratom_freq)
      error->all(FLERR,"Dump mozi and fix not computed at compatible times");
  }

  int ivariable;
  for (int i = 0; i < nvariable; i++) {
    ivariable = input->variable->find(id_variable[i]);
    if (ivariable < 0)
      error->all(FLERR,"Could not find dump mozi variable name");
    variable[i] = ivariable;
  }

  // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for dump mozi does not exist");
  }

  // open single file, one time only

  if (multifile == 0) openfile();
}

/* ---------------------------------------------------------------------- */

void DumpMozi::write_header(bigint ndump)
{
  const char* head1 = "#type 6 4 8 -1\n";
  const char* headend = "#\n";
  const char* headstl = "#stl ";
  double d;
  int a;
  d = static_cast<double>(update->ntimestep);
  if (multiproc) {
    if (ftell(fp) == 0) {
      fwrite(head1,strlen(head1),1,fp);
      if (stl_num > 0) {
	for (a=0;a<stl_num;a++) {
	  fwrite(headstl,strlen(headstl),1,fp);
	  fwrite(stl[a],strlen(stl[a]),1,fp);
	  fputc('\n',fp);
	}
      }
      fwrite(headend,strlen(headend),1,fp);
    }
    if (ndump > 0) fwrite(&d,D,1,fp);
  } else {
    if (me == 0) {
      if (ftell(fp) == 0) {
	fwrite(head1,strlen(head1),1,fp);
	if (stl_num > 0) {
	  for (a=0;a<stl_num;a++) {
	    fwrite(headstl,strlen(headstl),1,fp);
	    fwrite(stl[a],strlen(stl[a]),1,fp);
	    fputc('\n',fp);
	  }
	}
	fwrite(headend,strlen(headend),1,fp);
      }
      if (ndump > 0) fwrite(&d,D,1,fp);
    }
  }
}

/* ---------------------------------------------------------------------- */

int DumpMozi::count()
{
  int i,a;
  int sumnc;
  int maxnc;

  // grow choose and variable vbuf arrays if needed

  int nlocal = atom->nlocal;
  if (nlocal > maxlocal) {
    maxlocal = atom->nmax;

    memory->destroy(choose);
    memory->destroy(dchoose);
    memory->destroy(clist);
    memory->create(choose,maxlocal,"dump:choose");
    memory->create(dchoose,maxlocal,"dump:dchoose");
    memory->create(clist,maxlocal,"dump:clist");

    for (i = 0; i < nvariable; i++) {
      memory->destroy(vbuf[i]);
      memory->create(vbuf[i],maxlocal,"dump:vbuf");
    }
  }

  // invoke Computes for per-atom quantities

  if (ncompute) {
    for (i = 0; i < ncompute; i++)
      if (!(compute[i]->invoked_flag & INVOKED_PERATOM)) {
        compute[i]->compute_peratom();
        compute[i]->invoked_flag |= INVOKED_PERATOM;
      }
  }

  // evaluate atom-style Variables for per-atom quantities

  if (nvariable)
    for (i = 0; i < nvariable; i++)
      input->variable->compute_atom(variable[i],igroup,vbuf[i],1,0);

  // choose all local atoms for output

  for (i = 0; i < nlocal; i++) choose[i] = 1;

  // un-choose if not in group

  if (igroup) {
    int *mask = atom->mask;
    for (i = 0; i < nlocal; i++)
      if (!(mask[i] & groupbit))
        choose[i] = 0;
  }

  // un-choose if not in region

  if (iregion >= 0) {
    Region *region = domain->regions[iregion];
    double **x = atom->x;
    for (i = 0; i < nlocal; i++)
      if (choose[i] && region->match(x[i][0],x[i][1],x[i][2]) == 0)
        choose[i] = 0;
  }

  // un-choose if any threshhold criterion isn't met

  if (nthresh) {
    double *ptr;
    double value;
    int nstride;
    int nlocal = atom->nlocal;

    for (int ithresh = 0; ithresh < nthresh; ithresh++) {

      // customize by adding to if statement

      if (thresh_array[ithresh] == ID) {
        int *tag = atom->tag;
        for (i = 0; i < nlocal; i++) dchoose[i] = tag[i];
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == MOL) {
        if (!atom->molecule_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        int *molecule = atom->molecule;
        for (i = 0; i < nlocal; i++) dchoose[i] = molecule[i];
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == TYPE) {
        int *type = atom->type;
        for (i = 0; i < nlocal; i++) dchoose[i] = type[i];
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == ELEMENT) {
        int *type = atom->type;
        for (i = 0; i < nlocal; i++) dchoose[i] = type[i];
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == MASS) {
        if (atom->rmass) {
          ptr = atom->rmass;
          nstride = 1;
        } else {
          double *mass = atom->mass;
          int *type = atom->type;
          for (i = 0; i < nlocal; i++) dchoose[i] = mass[type[i]];
          ptr = dchoose;
          nstride = 1;
        }

      } else if (thresh_array[ithresh] == X) {
        ptr = &atom->x[0][0];
        nstride = 3;
      } else if (thresh_array[ithresh] == Y) {
        ptr = &atom->x[0][1];
        nstride = 3;
      } else if (thresh_array[ithresh] == Z) {
        ptr = &atom->x[0][2];
        nstride = 3;

      } else if (thresh_array[ithresh] == XS) {
        double **x = atom->x;
        double boxxlo = domain->boxlo[0];
        double invxprd = 1.0/domain->xprd;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = (x[i][0] - boxxlo) * invxprd;
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == YS) {
        double **x = atom->x;
        double boxylo = domain->boxlo[1];
        double invyprd = 1.0/domain->yprd;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = (x[i][1] - boxylo) * invyprd;
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == ZS) {
        double **x = atom->x;
        double boxzlo = domain->boxlo[2];
        double invzprd = 1.0/domain->zprd;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = (x[i][2] - boxzlo) * invzprd;
        ptr = dchoose;
        nstride = 1;

      } else if (thresh_array[ithresh] == XSTRI) {
        double **x = atom->x;
        double *boxlo = domain->boxlo;
        double *h_inv = domain->h_inv;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = h_inv[0]*(x[i][0]-boxlo[0]) +
            h_inv[5]*(x[i][1]-boxlo[1]) + h_inv[4]*(x[i][2]-boxlo[2]);
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == YSTRI) {
        double **x = atom->x;
        double *boxlo = domain->boxlo;
        double *h_inv = domain->h_inv;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = h_inv[1]*(x[i][1]-boxlo[1]) +
            h_inv[3]*(x[i][2]-boxlo[2]);
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == ZSTRI) {
        double **x = atom->x;
        double *boxlo = domain->boxlo;
        double *h_inv = domain->h_inv;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = h_inv[2]*(x[i][2]-boxlo[2]);
        ptr = dchoose;
        nstride = 1;

      } else if (thresh_array[ithresh] == XU) {
        double **x = atom->x;
        tagint *image = atom->image;
        double xprd = domain->xprd;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = x[i][0] + ((image[i] & IMGMASK) - IMGMAX) * xprd;
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == YU) {
        double **x = atom->x;
        tagint *image = atom->image;
        double yprd = domain->yprd;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = x[i][1] +
            ((image[i] >> IMGBITS & IMGMASK) - IMGMAX) * yprd;
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == ZU) {
        double **x = atom->x;
        tagint *image = atom->image;
        double zprd = domain->zprd;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = x[i][2] + ((image[i] >> IMG2BITS) - IMGMAX) * zprd;
        ptr = dchoose;
        nstride = 1;

      } else if (thresh_array[ithresh] == XUTRI) {
        double **x = atom->x;
        tagint *image = atom->image;
        double *h = domain->h;
        int xbox,ybox,zbox;
        for (i = 0; i < nlocal; i++) {
          xbox = (image[i] & IMGMASK) - IMGMAX;
          ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
          zbox = (image[i] >> IMG2BITS) - IMGMAX;
          dchoose[i] = x[i][0] + h[0]*xbox + h[5]*ybox + h[4]*zbox;
        }
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == YUTRI) {
        double **x = atom->x;
        tagint *image = atom->image;
        double *h = domain->h;
        int ybox,zbox;
        for (i = 0; i < nlocal; i++) {
          ybox = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
          zbox = (image[i] >> IMG2BITS) - IMGMAX;
          dchoose[i] = x[i][1] + h[1]*ybox + h[3]*zbox;
        }
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == ZUTRI) {
        double **x = atom->x;
        tagint *image = atom->image;
        double *h = domain->h;
        int zbox;
        for (i = 0; i < nlocal; i++) {
          zbox = (image[i] >> IMG2BITS) - IMGMAX;
          dchoose[i] = x[i][2] + h[2]*zbox;
        }
        ptr = dchoose;
        nstride = 1;

      } else if (thresh_array[ithresh] == XSU) {
        double **x = atom->x;
        tagint *image = atom->image;
        double boxxlo = domain->boxlo[0];
        double invxprd = 1.0/domain->xprd;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = (x[i][0] - boxxlo) * invxprd +
            (image[i] & IMGMASK) - IMGMAX;
        ptr = dchoose;
        nstride = 1;

      } else if (thresh_array[ithresh] == YSU) {
        double **x = atom->x;
        tagint *image = atom->image;
        double boxylo = domain->boxlo[1];
        double invyprd = 1.0/domain->yprd;
        for (i = 0; i < nlocal; i++)
          dchoose[i] =
            (x[i][1] - boxylo) * invyprd +
            (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
        ptr = dchoose;
        nstride = 1;

      } else if (thresh_array[ithresh] == ZSU) {
        double **x = atom->x;
        tagint *image = atom->image;
        double boxzlo = domain->boxlo[2];
        double invzprd = 1.0/domain->zprd;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = (x[i][2] - boxzlo) * invzprd +
            (image[i] >> IMG2BITS) - IMGMAX;
        ptr = dchoose;
        nstride = 1;

      } else if (thresh_array[ithresh] == XSUTRI) {
        double **x = atom->x;
        tagint *image = atom->image;
        double *boxlo = domain->boxlo;
        double *h_inv = domain->h_inv;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = h_inv[0]*(x[i][0]-boxlo[0]) +
            h_inv[5]*(x[i][1]-boxlo[1]) +
            h_inv[4]*(x[i][2]-boxlo[2]) +
            (image[i] & IMGMASK) - IMGMAX;
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == YSUTRI) {
        double **x = atom->x;
        tagint *image = atom->image;
        double *boxlo = domain->boxlo;
        double *h_inv = domain->h_inv;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = h_inv[1]*(x[i][1]-boxlo[1]) +
            h_inv[3]*(x[i][2]-boxlo[2]) +
            (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == ZSUTRI) {
        double **x = atom->x;
        tagint *image = atom->image;
        double *boxlo = domain->boxlo;
        double *h_inv = domain->h_inv;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = h_inv[2]*(x[i][2]-boxlo[2]) +
            (image[i] >> IMG2BITS) - IMGMAX;
        ptr = dchoose;
        nstride = 1;

      } else if (thresh_array[ithresh] == IX) {
        tagint *image = atom->image;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = (image[i] & IMGMASK) - IMGMAX;
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == IY) {
        tagint *image = atom->image;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = (image[i] >> IMGBITS & IMGMASK) - IMGMAX;
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == IZ) {
        tagint *image = atom->image;
        for (i = 0; i < nlocal; i++)
          dchoose[i] = (image[i] >> IMG2BITS) - IMGMAX;
        ptr = dchoose;
        nstride = 1;

      } else if (thresh_array[ithresh] == VX) {
        ptr = &atom->v[0][0];
        nstride = 3;
      } else if (thresh_array[ithresh] == VY) {
        ptr = &atom->v[0][1];
        nstride = 3;
      } else if (thresh_array[ithresh] == VZ) {
        ptr = &atom->v[0][2];
        nstride = 3;
      } else if (thresh_array[ithresh] == FX) {
        ptr = &atom->f[0][0];
        nstride = 3;
      } else if (thresh_array[ithresh] == FY) {
        ptr = &atom->f[0][1];
        nstride = 3;
      } else if (thresh_array[ithresh] == FZ) {
        ptr = &atom->f[0][2];
        nstride = 3;

      } else if (thresh_array[ithresh] == Q) {
        if (!atom->q_flag)
          error->all(FLERR,"Threshhold for an atom property that isn't allocated");
        ptr = atom->q;
        nstride = 1;
      } else if (thresh_array[ithresh] == P) { 
        if (!atom->p_flag)
          error->all(FLERR,"Threshhold for an atom property that isn't allocated");
        ptr = atom->p;
        nstride = 1;
      } else if (thresh_array[ithresh] == RHO) { 
        if (!atom->rho_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = atom->rho;
        nstride = 1;
      } else if (thresh_array[ithresh] == DENSITY) { 
        if (!atom->density_flag)
          error->all(FLERR,"Threshhold for an atom property that isn't allocated");
        ptr = atom->density;
        nstride = 1;
      } else if (thresh_array[ithresh] == MUX) {
        if (!atom->mu_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->mu[0][0];
        nstride = 4;
      } else if (thresh_array[ithresh] == MUY) {
        if (!atom->mu_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->mu[0][1];
        nstride = 4;
      } else if (thresh_array[ithresh] == MUZ) {
        if (!atom->mu_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->mu[0][2];
        nstride = 4;
      } else if (thresh_array[ithresh] == MU) {
        if (!atom->mu_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->mu[0][3];
        nstride = 4;

      } else if (thresh_array[ithresh] == RADIUS) {
        if (!atom->radius_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = atom->radius;
        nstride = 1;
      } else if (thresh_array[ithresh] == DIAMETER) {
        if (!atom->radius_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        double *radius = atom->radius;
        for (i = 0; i < nlocal; i++) dchoose[i] = 2.0*radius[i];
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == OMEGAX) {
        if (!atom->omega_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->omega[0][0];
        nstride = 3;
      } else if (thresh_array[ithresh] == OMEGAY) {
        if (!atom->omega_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->omega[0][1];
        nstride = 3;
      } else if (thresh_array[ithresh] == OMEGAZ) {
        if (!atom->omega_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->omega[0][2];
        nstride = 3;
      } else if (thresh_array[ithresh] == ANGMOMX) {
        if (!atom->angmom_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->angmom[0][0];
        nstride = 3;
      } else if (thresh_array[ithresh] == ANGMOMY) {
        if (!atom->angmom_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->angmom[0][1];
        nstride = 3;
      } else if (thresh_array[ithresh] == ANGMOMZ) {
        if (!atom->angmom_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->angmom[0][2];
        nstride = 3;
      } else if (thresh_array[ithresh] == TQX) {
        if (!atom->torque_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->torque[0][0];
        nstride = 3;
      } else if (thresh_array[ithresh] == TQY) {
        if (!atom->torque_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->torque[0][1];
        nstride = 3;
      } else if (thresh_array[ithresh] == TQZ) {
        if (!atom->torque_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = &atom->torque[0][2];
        nstride = 3;

      } else if (thresh_array[ithresh] == SPIN) {
        if (!atom->spin_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        int *spin = atom->spin;
        for (i = 0; i < nlocal; i++) dchoose[i] = spin[i];
        ptr = dchoose;
        nstride = 1;
      } else if (thresh_array[ithresh] == ERADIUS) {
        if (!atom->eradius_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = atom->eradius;
        nstride = 1;
      } else if (thresh_array[ithresh] == ERVEL) {
        if (!atom->ervel_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = atom->ervel;
        nstride = 1;
      } else if (thresh_array[ithresh] == ERFORCE) {
        if (!atom->erforce_flag)
          error->all(FLERR,
                     "Threshhold for an atom property that isn't allocated");
        ptr = atom->erforce;
        nstride = 1;

      } else if (thresh_array[ithresh] == COMPUTE) {
        i = nfield + ithresh;
        if (argindex[i] == 0) {
          ptr = compute[field2index[i]]->vector_atom;
          nstride = 1;
        } else {
          ptr = &compute[field2index[i]]->array_atom[0][argindex[i]-1];
          nstride = compute[field2index[i]]->size_peratom_cols;
        }

      } else if (thresh_array[ithresh] == FIX) {
        i = nfield + ithresh;
        if (argindex[i] == 0) {
          ptr = fix[field2index[i]]->vector_atom;
          nstride = 1;
        } else {
          ptr = &fix[field2index[i]]->array_atom[0][argindex[i]-1];
          nstride = fix[field2index[i]]->size_peratom_cols;
        }

      } else if (thresh_array[ithresh] == VARIABLE) {
        i = nfield + ithresh;
        ptr = vbuf[field2index[i]];
        nstride = 1;
      }

      // unselect atoms that don't meet threshhold criterion

      value = thresh_value[ithresh];

      if (thresh_op[ithresh] == LT) {
        for (i = 0; i < nlocal; i++, ptr += nstride)
          if (choose[i] && *ptr >= value) choose[i] = 0;
      } else if (thresh_op[ithresh] == LE) {
        for (i = 0; i < nlocal; i++, ptr += nstride)
          if (choose[i] && *ptr > value) choose[i] = 0;
      } else if (thresh_op[ithresh] == GT) {
        for (i = 0; i < nlocal; i++, ptr += nstride)
          if (choose[i] && *ptr <= value) choose[i] = 0;
      } else if (thresh_op[ithresh] == GE) {
        for (i = 0; i < nlocal; i++, ptr += nstride)
          if (choose[i] && *ptr < value) choose[i] = 0;
      } else if (thresh_op[ithresh] == EQ) {
        for (i = 0; i < nlocal; i++, ptr += nstride)
          if (choose[i] && *ptr != value) choose[i] = 0;
      } else if (thresh_op[ithresh] == NEQ) {
        for (i = 0; i < nlocal; i++, ptr += nstride)
          if (choose[i] && *ptr == value) choose[i] = 0;
      }
    }
  }

  // compress choose flags into clist
  // nchoose = # of selected atoms
  // clist[i] = local index of each selected atom

  nchoose = 0;
  for (i = 0; i < nlocal; i++)
    if (choose[i]) clist[nchoose++] = i;

  if (haveforces) {
    /* Calculate contact forces */
    forces->compute_local();
/*    nc = forces->get_ncount();
    MPI_Allreduce(&nc,&sumnc,1,MPI_INT,MPI_SUM,world);
    MPI_Allreduce(&nc,&maxnc,1,MPI_INT,MPI_MAX,world);
    if (sumnc + 1 > save_cont_size) {
      save_cont_size = sumnc + 1;
      savecontbuff = (double *)memory->srealloc(savecontbuff,
	  forces->get_nvalues()*save_cont_size*
	  sizeof(double),"dump:savecontbuff");
    }
    if (maxnc + 1 > cont_size) {
      cont_size = maxnc + 1;
      contbuff = (double *)memory->srealloc(contbuff,
	  forces->get_nvalues()*cont_size*sizeof(double),"dump:contbuff");
    }*/
  }
  for (a=0;a<wall_num;a++) {
    /* Calculate contact forces */
    wall[a]->compute_local();
/*    nc = wall[a]->get_ncount();
    MPI_Allreduce(&nc,&sumnc,1,MPI_INT,MPI_SUM,world);
    MPI_Allreduce(&nc,&maxnc,1,MPI_INT,MPI_MAX,world);
    if (sumnc + 1 > save_cont_size) {
      save_cont_size = sumnc + 1;
      savecontbuff = (double *)memory->srealloc(savecontbuff,
	  wall[a]->get_nvalues()*save_cont_size*
	  sizeof(double),"dump:savecontbuff");
    }
    if (maxnc + 1 > cont_size) {
      cont_size = maxnc + 1;
      contbuff = (double *)memory->srealloc(contbuff,
	  wall[a]->get_nvalues()*cont_size*sizeof(double),"dump:contbuff");
    }*/
  }
  return nchoose;
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack(int *ids)
{
  for (int n = 0; n < size_one; n++) (this->*pack_choice[n])(n);
  if (ids) {
    int *tag = atom->tag;
    for (int i = 0; i < nchoose; i++)
      ids[i] = tag[clist[i]];
  }
}

/* ----------------------------------------------------------------------
   convert mybuf of doubles to one big formatted string in sbuf
   return -1 if strlen exceeds an int, since used as arg in MPI calls in Dump
------------------------------------------------------------------------- */

int DumpMozi::convert_string(int n, double *mybuf)
{
  int i,j;

  int offset = 0;
  int m = 0;
  for (i = 0; i < n; i++) {
    if (offset + size_one*ONEFIELD > maxsbuf) {
      if ((bigint) maxsbuf + DELTA > MAXSMALLINT) return -1;
      maxsbuf += DELTA;
      memory->grow(sbuf,maxsbuf,"dump:sbuf");
    }

    for (j = 0; j < size_one; j++) {
      if (vtype[j] == INT)
        offset += sprintf(&sbuf[offset],vformat[j],static_cast<int> (mybuf[m]));
      else if (vtype[j] == DOUBLE)
        offset += sprintf(&sbuf[offset],vformat[j],mybuf[m]);
      else if (vtype[j] == STRING)
        offset += sprintf(&sbuf[offset],vformat[j],typenames[(int) mybuf[m]]);
      m++;
    }
    offset += sprintf(&sbuf[offset],"\n");
  }

  return offset;
}

/* ---------------------------------------------------------------------- */

void DumpMozi::write_data(int n, double *mybuf)
{
  (this->*write_choice)(n,mybuf);
}

/* ---------------------------------------------------------------------- */

int compare_savedatom(const void *a, const void *b)
{
  double *p1,*p2, *t1,*t2;
  int typ1,typ2,tag1,tag2;

  p1 = (double *)a;
  p2 = (double *)b;
  t1 = p1 + 1; t2 = p2 + 1;
  typ1 = static_cast<int>(*p1);
  typ2 = static_cast<int>(*p2);
  tag1 = static_cast<int>(*t1);
  tag2 = static_cast<int>(*t2);
  if (typ1 < typ2) return -1;
  else if (typ1 == typ2) {
    if (tag1 < tag2) return -1;
    else if (tag1 == tag2) return 0;
    else return 1;
  } else return 1;
}

/* ---------------------------------------------------------------------- */

void DumpMozi::write_binary(int n, double *mybuf)
{
  int a,b,i,enda;
  int fnc;
  double d;
  double *x1,*x2,*f,dx[3],L[3],rx[4];
  double **array;

  if (n == 0) return;
  qsort(mybuf, n, size_one*sizeof(double), compare_savedatom);
  for (a = 0; a < n;) {
    enda = a + 1;
    while ((enda < n) && (mybuf[a*size_one] == mybuf[enda*size_one])) enda++;
    fwrite(&btype,I,1,fp);
    i = 0xffffff00U+((static_cast<int> (mybuf[a*size_one]) + 1)&15);
    fwrite(&i,I,1,fp);
    i = enda - a;
    fwrite(&i,I,1,fp);
    for (;a<enda;a++) {
      if (!atom->superquadric_flag) d = mybuf[a * size_one + 2] * 2.0;
      else d = mybuf[a * size_one + 2];
      fwrite(&d,D,1,fp);
      if (angle_pos) {
	fwrite(&mybuf[a * size_one + 3],D,angle_pos - 3,fp);
	d = mybuf[a * size_one + angle_pos];
	if (fabs(d) <= 1.0) d = acos(d);
	else d = 0.0;
/*	printf("RRR1 %lg %lg %lg\n",mybuf[a * size_one + 2],
	  mybuf[a * size_one + 3],mybuf[a * size_one + 4]);
	printf("RRR2 %lg %lg %lg\n",mybuf[a * size_one + 5],
	  mybuf[a * size_one + 6],mybuf[a * size_one + 7]);
	printf("RRR3 %lg %lg %lg %lg\n",mybuf[a * size_one + 8],
	  mybuf[a * size_one + 9],mybuf[a * size_one + 10],
	  mybuf[a * size_one + 11]);
	printf("RRR4 %lg %lg %lg\n",mybuf[a * size_one + 12],
	  mybuf[a * size_one + 13],mybuf[a * size_one + 14]);*/
/*	printf("QQQ %d   %lg %lg %lg %lg\n",angle_pos,
	    mybuf[a * size_one + angle_pos - 3],
	    mybuf[a * size_one + angle_pos - 2],
	    mybuf[a * size_one + angle_pos - 1],
	    mybuf[a * size_one + angle_pos]);*/
	fwrite(&d,D,1,fp);
	fwrite(&mybuf[a * size_one + angle_pos + 1],D,
	  size_one - angle_pos - 1,fp);
      } else {
	fwrite(&mybuf[a * size_one + 2],D,size_one - 2,fp);
      }
      i = static_cast<int>(mybuf[a * size_one + 1]);
      fwrite(&i,I,1,fp);
    }
  }
  if (haveforces) {
    fnc = forces->get_ncount();
    array = forces->array_local;
    /* BP_Pos+BP_Orient+BT_Force */
    i=6+32+64+512;
    fwrite(&i,I,1,fp);
    /* print out texture, actually red */
    i=0x0000ff00U;
    fwrite(&i,I,1,fp);
    /* number */
    fwrite(&fnc,I,1,fp);
    /* data */
    for (a=0;a<fnc;a++) {
      x1 = &array[a][0];
      x2 = &array[a][3];
      f = &array[a][9];
      d = sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
      /* width of cylinder */
      fwrite(&d,D,1,fp);
      for (b=0;b<3;b++) {
	dx[b] = x2[b]-x1[b];
	L[b] = domain->boxhi[b] - domain->boxlo[b];
	dx[b] = (fabs(dx[b]) > (0.5*L[b]) ? (dx[b] - L[b]*SIGN(dx[b])) : dx[b]);
      }
      d = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
      for (b=0;b<3;b++) dx[b] /= d;
      /* length of cylinder */
      fwrite(&d,D,1,fp);
      /* indices */
      i = static_cast<int>(array[a][6]-1.0);
      fwrite(&i,I,1,fp);
      i = static_cast<int>(array[a][7])-1.0;
      fwrite(&i,I,1,fp);
      /* position */
      fwrite(x1,D,3,fp);
      /* orient */
      rx[0] = dx[1];
      rx[1] = -dx[0];
      rx[2] = 0.0;
      if ((rx[0] == 0.0) && (rx[1] == 0.0)) {
	rx[0] = 1.0;
	rx[3] = 0.0;
      } else {
	rx[3] = -asin(MIN(sqrt(rx[0]*rx[0]+rx[1]*rx[1]),1.0));
      }
      fwrite(rx,D,4,fp);
      /* force */
      fwrite(f,D,3,fp);
    }
  }
  for (a=0;a<wall_num;a++) {
    fnc = wall[a]->get_ncount();
    array = wall[a]->array_local;
    /* BP_Pos+BP_Orient+BT_Force */
    i=6+32+64+512;
    fwrite(&i,I,1,fp);
    /* print out texture, actually red */
    i=0x0000ff00U;
    fwrite(&i,I,1,fp);
    /* number */
    fwrite(&fnc,I,1,fp);
    /* data */
    for (a=0;a<fnc;a++) {
      x1 = &array[a][0];
      x2 = &array[a][3];
      f = &array[a][9];
      d = sqrt(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
      /* width of cylinder */
      fwrite(&d,D,1,fp);
      for (b=0;b<3;b++) {
	dx[b] = x2[b]-x1[b];
	L[b] = domain->boxhi[b] - domain->boxlo[b];
	dx[b] = (fabs(dx[b]) > (0.5*L[b]) ? (dx[b] - L[b]*SIGN(dx[b])) : dx[b]);
      }
      d = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
      for (b=0;b<3;b++) dx[b] /= d;
      /* length of cylinder */
      fwrite(&d,D,1,fp);
      /* indices */
      i = static_cast<int>(array[a][8]-1.0);
      fwrite(&i,I,1,fp);
      i = -1 - a;
      fwrite(&i,I,1,fp);
      /* position */
      fwrite(x1,D,3,fp);
      /* orient */
      rx[0] = dx[1];
      rx[1] = -dx[0];
      rx[2] = 0.0;
      if ((rx[0] == 0.0) && (rx[1] == 0.0)) {
	rx[0] = 1.0;
	rx[3] = 0.0;
      } else {
	rx[3] = -asin(MIN(sqrt(rx[0]*rx[0]+rx[1]*rx[1]),1.0));
      }
      fwrite(rx,D,4,fp);
      /* force */
      fwrite(f,D,3,fp);
    }
  }
  i = 0;
  fwrite(&i,I,1,fp);
}

/* ---------------------------------------------------------------------- */

void DumpMozi::write_string(int n, double *mybuf)
{
  fwrite(mybuf,sizeof(char),n,fp);
}

/* ---------------------------------------------------------------------- */

void DumpMozi::write_lines(int n, double *mybuf)
{
  int i,j;

  int m = 0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < size_one; j++) {
      if (vtype[j] == INT) fprintf(fp,vformat[j],static_cast<int> (mybuf[m]));
      else if (vtype[j] == DOUBLE) fprintf(fp,vformat[j],mybuf[m]);
      else if (vtype[j] == STRING)
        fprintf(fp,vformat[j],typenames[(int) mybuf[m]]);
      m++;
    }
    fprintf(fp,"\n");
  }
}

/* ---------------------------------------------------------------------- */

/* returns the number of fields to be stored */
int DumpMozi::pre_parse_fields(int narg, char **arg)
{
  int iarg,a;
  int field_num=3; /* radius and type must be there */
  char errmsg[100];

  stl_num = 0;
  if (!atom->superquadric_flag) {
    btype = 2; /* BT_Ball */
  } else {
    btype = 17; /* BT_Quadratic */
    field_num += 2; /* For shape y and z */
  }
  btype += 32768; /* BP_Handle */
  for (iarg = 5; iarg < narg; iarg++) {
    a=0;
    while (strlen(parnam[a])) {
      if (strcmp(arg[iarg],parnam[a])==0) break;
      a++;
    }
    if (strlen(parnam[a])==0) {
      if (strcmp(arg[iarg],"ContactForces") == 0) {
	iarg++;
      } else if (strcmp(arg[iarg],"stl") == 0) {
	stl_num++; iarg++;
      } else if (strcmp(arg[iarg],"Wall") == 0) {
	wall_num++; iarg++;
      } else {
	sprintf(errmsg,"Dump mozi does not know about %s",arg[iarg]);
	error->all(FLERR, errmsg);
      }
    } else {
      field_num+=parlength[a];
      btype |= 1<<(a+5); /* Pos is 32 */
    }
  }
  return(field_num);
}

/* ---------------------------------------------------------------------- */

int DumpMozi::parse_fields(int narg, char **arg)
{
  // customize by adding to if statement

  int i,n;
  i = 0;
  posi = 0;
  stl_num = 0;
  wall_num = 0;
  angle_pos = 0;
  pack_choice[i] = &DumpMozi::pack_type;
  vtype[i] = INT;
  i++;
  pack_choice[i] = &DumpMozi::pack_id;
  vtype[i] = INT;
  i++;
  if (!atom->superquadric_flag) {
    pack_choice[i] = &DumpMozi::pack_radius;
    vtype[i] = DOUBLE;
    i++;
  } else {
    pack_choice[i] = &DumpMozi::pack_shapex;
    vtype[i] = DOUBLE;
    i++;
    pack_choice[i] = &DumpMozi::pack_shapey;
    vtype[i] = DOUBLE;
    i++;
    pack_choice[i] = &DumpMozi::pack_shapez;
    vtype[i] = DOUBLE;
    i++;
  }
  for (int iarg = 5; iarg < narg; iarg++) {
    if (strcmp(arg[iarg],"Pos") == 0) {
      posi = i;
      pack_choice[i] = &DumpMozi::pack_x;
      vtype[i] = DOUBLE;
      i++;
      pack_choice[i] = &DumpMozi::pack_y;
      vtype[i] = DOUBLE;
      i++;
      pack_choice[i] = &DumpMozi::pack_z;
      vtype[i] = DOUBLE;
      i++;
    } else if (strcmp(arg[iarg],"Orient") == 0) {
      if (!atom->superquadric_flag)
	error->all(FLERR,"Dumping Orientation requires superquadratic");
      pack_choice[i] = &DumpMozi::pack_quat2;
      vtype[i] = ORIENT;
      i++;
      pack_choice[i] = &DumpMozi::pack_quat3;
      vtype[i] = ORIENT;
      i++;
      pack_choice[i] = &DumpMozi::pack_quat4;
      vtype[i] = ORIENT;
      i++;
      angle_pos = i;
      pack_choice[i] = &DumpMozi::pack_quat1;
      vtype[i] = ORIENT;
      i++;
    } else if (strcmp(arg[iarg],"Coord") == 0) {
      error->all(FLERR,"Coordintation number is not yet supported");
      /* Check if list->numneigh[i] is available if neigh_list.h is
       * included! */
/*      pack_choice[i] = &DumpMozi::pack_coord;*/
      vtype[i] = INT;
      i++;
    } else if (strcmp(arg[iarg],"Velo") == 0) {
      pack_choice[i] = &DumpMozi::pack_vx;
      vtype[i] = DOUBLE;
      i++;
      pack_choice[i] = &DumpMozi::pack_vy;
      vtype[i] = DOUBLE;
      i++;
      pack_choice[i] = &DumpMozi::pack_vz;
      vtype[i] = DOUBLE;
      i++;
    } else if (strcmp(arg[iarg],"Force") == 0) {
      pack_choice[i] = &DumpMozi::pack_fx;
      vtype[i] = DOUBLE;
      i++;
      pack_choice[i] = &DumpMozi::pack_fy;
      vtype[i] = DOUBLE;
      i++;
      pack_choice[i] = &DumpMozi::pack_fz;
      vtype[i] = DOUBLE;
      i++;
    } else if (strcmp(arg[iarg],"AngVelo") == 0) {
      if (!atom->omega_flag)
	error->all(FLERR,"Dumping omega which isn't allocated");
      pack_choice[i] = &DumpMozi::pack_omegax;
      vtype[i] = DOUBLE;
      i++;
      pack_choice[i] = &DumpMozi::pack_omegay;
      vtype[i] = DOUBLE;
      i++;
      pack_choice[i] = &DumpMozi::pack_omegaz;
      vtype[i] = DOUBLE;
      i++;
    } else if (strcmp(arg[iarg],"Torque") == 0) {
      if (!atom->torque_flag)
	error->all(FLERR,"Dumping torque which isn't allocated");
      pack_choice[i] = &DumpMozi::pack_tqx;
      vtype[i] = DOUBLE;
      i++;
      pack_choice[i] = &DumpMozi::pack_tqy;
      vtype[i] = DOUBLE;
      i++;
      pack_choice[i] = &DumpMozi::pack_tqz;
      vtype[i] = DOUBLE;
      i++;
    } else if (strcmp(arg[iarg],"ContactForces") == 0) {
      if (posi == 0) error->all(FLERR,"ContactForces require Pos atom");
      iarg++;
      if (strncmp(arg[iarg],"c_",2) != 0)
	error->all(FLERR,"ContactForces can only be contactforces compute");

      n = modify->find_compute(&arg[iarg][2]);
      if (n < 0) error->all(FLERR,"Could not find dump mozi CF compute ID");
      if (modify->compute[n]->local_flag == 0)
	error->all(FLERR,"Dump mozi CF compute ID does not compute local info");
      forces = static_cast<ComputePairGranLocal*>(modify->compute[n]);
      haveforces = 1;
    } else if (strcmp(arg[iarg],"stl") == 0) {
      iarg++;
      stl[stl_num++] = strdup(arg[iarg]);
    } else if (strcmp(arg[iarg],"Wall") == 0) {
      iarg++;
      if (strncmp(arg[iarg],"c_",2) != 0)
	error->all(FLERR,"Walls must be computes");

      n = modify->find_compute(&arg[iarg][2]);
      if (n < 0) error->all(FLERR,"Could not find dump mozi wall fix ID");
      wall[wall_num++] =
	static_cast<ComputePairGranLocal*>(modify->compute[n]);
    }
  }
  return(i);
}

/* ----------------------------------------------------------------------
   add Compute to list of Compute objects used by dump
   return index of where this Compute is in list
   if already in list, do not add, just return index, else add to list
------------------------------------------------------------------------- */

int DumpMozi::add_compute(char *id)
{
  int icompute;
  for (icompute = 0; icompute < ncompute; icompute++)
    if (strcmp(id,id_compute[icompute]) == 0) break;
  if (icompute < ncompute) return icompute;

  id_compute = (char **)
    memory->srealloc(id_compute,(ncompute+1)*sizeof(char *),"dump:id_compute");
  delete [] compute;
  compute = new Compute*[ncompute+1];

  int n = strlen(id) + 1;
  id_compute[ncompute] = new char[n];
  strcpy(id_compute[ncompute],id);
  ncompute++;
  return ncompute-1;
}

/* ----------------------------------------------------------------------
   add Fix to list of Fix objects used by dump
   return index of where this Fix is in list
   if already in list, do not add, just return index, else add to list
------------------------------------------------------------------------- */

int DumpMozi::add_fix(char *id)
{
  int ifix;
  for (ifix = 0; ifix < nfix; ifix++)
    if (strcmp(id,id_fix[ifix]) == 0) break;
  if (ifix < nfix) return ifix;

  id_fix = (char **)
    memory->srealloc(id_fix,(nfix+1)*sizeof(char *),"dump:id_fix");
  delete [] fix;
  fix = new Fix*[nfix+1];

  int n = strlen(id) + 1;
  id_fix[nfix] = new char[n];
  strcpy(id_fix[nfix],id);
  nfix++;
  return nfix-1;
}

/* ----------------------------------------------------------------------
   add Variable to list of Variables used by dump
   return index of where this Variable is in list
   if already in list, do not add, just return index, else add to list
------------------------------------------------------------------------- */

int DumpMozi::add_variable(char *id)
{
  int ivariable;
  for (ivariable = 0; ivariable < nvariable; ivariable++)
    if (strcmp(id,id_variable[ivariable]) == 0) break;
  if (ivariable < nvariable) return ivariable;

  id_variable = (char **)
    memory->srealloc(id_variable,(nvariable+1)*sizeof(char *),
                     "dump:id_variable");
  delete [] variable;
  variable = new int[nvariable+1];
  delete [] vbuf;
  vbuf = new double*[nvariable+1];
  for (int i = 0; i <= nvariable; i++) vbuf[i] = NULL;

  int n = strlen(id) + 1;
  id_variable[nvariable] = new char[n];
  strcpy(id_variable[nvariable],id);
  nvariable++;
  return nvariable-1;
}

/* ---------------------------------------------------------------------- */

int DumpMozi::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"region") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    if (strcmp(arg[1],"none") == 0) iregion = -1;
    else {
      iregion = domain->find_region(arg[1]);
      if (iregion == -1)
        error->all(FLERR,"Dump_modify region ID does not exist");
      delete [] idregion;
      int n = strlen(arg[1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[1]);
    }
    return 2;
  }

  if (strcmp(arg[0],"label") == 0) { 
     if (narg < 2) error->all(FLERR,"Illegal dump_modify command [label]");
     delete [] label;
     int n = strlen(arg[1]) + 1;
     label = new char[n];
     strcpy(label,arg[1]);
     return 2;
   }

  if (strcmp(arg[0],"element") == 0) {
    if (narg < ntypes+1)
      error->all(FLERR,"Dump modify element names do not match atom types");

    if (typenames) {
      for (int i = 1; i <= ntypes; i++) delete [] typenames[i];
      delete [] typenames;
      typenames = NULL;
    }

    typenames = new char*[ntypes+1];
    for (int itype = 1; itype <= ntypes; itype++) {
      int n = strlen(arg[itype]) + 1;
      typenames[itype] = new char[n];
      strcpy(typenames[itype],arg[itype]);
    }
    return ntypes+1;
  }

  if (strcmp(arg[0],"thresh") == 0) {
    if (narg < 2) error->all(FLERR,"Illegal dump_modify command");
    if (strcmp(arg[1],"none") == 0) {
      if (nthresh) {
        memory->destroy(thresh_array);
        memory->destroy(thresh_op);
        memory->destroy(thresh_value);
        thresh_array = NULL;
        thresh_op = NULL;
        thresh_value = NULL;
      }
      nthresh = 0;
      return 2;
    }

    if (narg < 4) error->all(FLERR,"Illegal dump_modify command");

    // grow threshhold arrays

    memory->grow(thresh_array,nthresh+1,"dump:thresh_array");
    memory->grow(thresh_op,(nthresh+1),"dump:thresh_op");
    memory->grow(thresh_value,(nthresh+1),"dump:thresh_value");

    // set attribute type of threshhold
    // customize by adding to if statement

    if (strcmp(arg[1],"id") == 0) thresh_array[nthresh] = ID;
    else if (strcmp(arg[1],"mol") == 0 || strcmp(arg[1],"id_multisphere") == 0) thresh_array[nthresh] = MOL;
    else if (strcmp(arg[1],"type") == 0) thresh_array[nthresh] = TYPE;
    else if (strcmp(arg[1],"mass") == 0) thresh_array[nthresh] = MASS;

    else if (strcmp(arg[1],"x") == 0) thresh_array[nthresh] = X;
    else if (strcmp(arg[1],"y") == 0) thresh_array[nthresh] = Y;
    else if (strcmp(arg[1],"z") == 0) thresh_array[nthresh] = Z;

    else if (strcmp(arg[1],"xs") == 0 && domain->triclinic == 0)
      thresh_array[nthresh] = XS;
    else if (strcmp(arg[1],"xs") == 0 && domain->triclinic == 1)
      thresh_array[nthresh] = XSTRI;
    else if (strcmp(arg[1],"ys") == 0 && domain->triclinic == 0)
      thresh_array[nthresh] = YS;
    else if (strcmp(arg[1],"ys") == 0 && domain->triclinic == 1)
      thresh_array[nthresh] = YSTRI;
    else if (strcmp(arg[1],"zs") == 0 && domain->triclinic == 0)
      thresh_array[nthresh] = ZS;
    else if (strcmp(arg[1],"zs") == 0 && domain->triclinic == 1)
      thresh_array[nthresh] = ZSTRI;

    else if (strcmp(arg[1],"xu") == 0 && domain->triclinic == 0)
      thresh_array[nthresh] = XU;
    else if (strcmp(arg[1],"xu") == 0 && domain->triclinic == 1)
      thresh_array[nthresh] = XUTRI;
    else if (strcmp(arg[1],"yu") == 0 && domain->triclinic == 0)
      thresh_array[nthresh] = YU;
    else if (strcmp(arg[1],"yu") == 0 && domain->triclinic == 1)
      thresh_array[nthresh] = YUTRI;
    else if (strcmp(arg[1],"zu") == 0 && domain->triclinic == 0)
      thresh_array[nthresh] = ZU;
    else if (strcmp(arg[1],"zu") == 0 && domain->triclinic == 1)
      thresh_array[nthresh] = ZUTRI;

    else if (strcmp(arg[1],"xsu") == 0 && domain->triclinic == 0)
      thresh_array[nthresh] = XSU;
    else if (strcmp(arg[1],"xsu") == 0 && domain->triclinic == 1)
      thresh_array[nthresh] = XSUTRI;
    else if (strcmp(arg[1],"ysu") == 0 && domain->triclinic == 0)
      thresh_array[nthresh] = YSU;
    else if (strcmp(arg[1],"ysu") == 0 && domain->triclinic == 1)
      thresh_array[nthresh] = YSUTRI;
    else if (strcmp(arg[1],"zsu") == 0 && domain->triclinic == 0)
      thresh_array[nthresh] = ZSU;
    else if (strcmp(arg[1],"zsu") == 0 && domain->triclinic == 1)
      thresh_array[nthresh] = ZSUTRI;

    else if (strcmp(arg[1],"ix") == 0) thresh_array[nthresh] = IX;
    else if (strcmp(arg[1],"iy") == 0) thresh_array[nthresh] = IY;
    else if (strcmp(arg[1],"iz") == 0) thresh_array[nthresh] = IZ;
    else if (strcmp(arg[1],"vx") == 0) thresh_array[nthresh] = VX;
    else if (strcmp(arg[1],"vy") == 0) thresh_array[nthresh] = VY;
    else if (strcmp(arg[1],"vz") == 0) thresh_array[nthresh] = VZ;
    else if (strcmp(arg[1],"fx") == 0) thresh_array[nthresh] = FX;
    else if (strcmp(arg[1],"fy") == 0) thresh_array[nthresh] = FY;
    else if (strcmp(arg[1],"fz") == 0) thresh_array[nthresh] = FZ;

    else if (strcmp(arg[1],"q") == 0) thresh_array[nthresh] = Q;
    else if (strcmp(arg[1],"density") == 0) thresh_array[nthresh] = DENSITY; 
    else if (strcmp(arg[1],"p") == 0) thresh_array[nthresh] = P; 
    else if (strcmp(arg[1],"rho") == 0) thresh_array[nthresh] = RHO; 
    else if (strcmp(arg[1],"mux") == 0) thresh_array[nthresh] = MUX;
    else if (strcmp(arg[1],"muy") == 0) thresh_array[nthresh] = MUY;
    else if (strcmp(arg[1],"muz") == 0) thresh_array[nthresh] = MUZ;
    else if (strcmp(arg[1],"mu") == 0) thresh_array[nthresh] = MU;

    else if (strcmp(arg[1],"radius") == 0) thresh_array[nthresh] = RADIUS;
    else if (strcmp(arg[1],"diameter") == 0) thresh_array[nthresh] = DIAMETER;
    else if (strcmp(arg[1],"omegax") == 0) thresh_array[nthresh] = OMEGAX;
    else if (strcmp(arg[1],"omegay") == 0) thresh_array[nthresh] = OMEGAY;
    else if (strcmp(arg[1],"omegaz") == 0) thresh_array[nthresh] = OMEGAZ;
    else if (strcmp(arg[1],"angmomx") == 0) thresh_array[nthresh] = ANGMOMX;
    else if (strcmp(arg[1],"angmomy") == 0) thresh_array[nthresh] = ANGMOMY;
    else if (strcmp(arg[1],"angmomz") == 0) thresh_array[nthresh] = ANGMOMZ;
    else if (strcmp(arg[1],"tqx") == 0) thresh_array[nthresh] = TQX;
    else if (strcmp(arg[1],"tqy") == 0) thresh_array[nthresh] = TQY;
    else if (strcmp(arg[1],"tqz") == 0) thresh_array[nthresh] = TQZ;

    else if (strcmp(arg[1],"spin") == 0) thresh_array[nthresh] = SPIN;
    else if (strcmp(arg[1],"eradius") == 0) thresh_array[nthresh] = ERADIUS;
    else if (strcmp(arg[1],"ervel") == 0) thresh_array[nthresh] = ERVEL;
    else if (strcmp(arg[1],"erforce") == 0) thresh_array[nthresh] = ERFORCE;
    else if (strcmp(arg[1],"shapex") == 0) thresh_array[nthresh] = SHAPEX; 
    else if (strcmp(arg[1],"shapey") == 0) thresh_array[nthresh] = SHAPEY;
    else if (strcmp(arg[1],"shapez") == 0) thresh_array[nthresh] = SHAPEZ;
    else if (strcmp(arg[1],"quat1") == 0) thresh_array[nthresh] = QUAT1;
    else if (strcmp(arg[1],"quat2") == 0) thresh_array[nthresh] = QUAT2;
    else if (strcmp(arg[1],"quat3") == 0) thresh_array[nthresh] = QUAT3;
    else if (strcmp(arg[1],"quat4") == 0) thresh_array[nthresh] = QUAT4;
    else if (strcmp(arg[1],"blockiness1") == 0) thresh_array[nthresh] = BLOCKINESS1;
    else if (strcmp(arg[1],"blockiness2") == 0) thresh_array[nthresh] = BLOCKINESS2;
    else if (strcmp(arg[1],"inertiax") == 0) thresh_array[nthresh] = INERTIAX;
    else if (strcmp(arg[1],"inertiay") == 0) thresh_array[nthresh] = INERTIAY;
    else if (strcmp(arg[1],"inertiaz") == 0) thresh_array[nthresh] = INERTIAZ;

    // compute value = c_ID
    // if no trailing [], then arg is set to 0, else arg is between []
    // must grow field2index and argindex arrays, since access is beyond nfield

    else if (strncmp(arg[1],"c_",2) == 0) {
      thresh_array[nthresh] = COMPUTE;
      memory->grow(field2index,nfield+nthresh+1,"dump:field2index");
      memory->grow(argindex,nfield+nthresh+1,"dump:argindex");
      int n = strlen(arg[1]);
      char *suffix = new char[n];
      strcpy(suffix,&arg[1][2]);

      char *ptr = strchr(suffix,'[');
      if (ptr) {
        if (suffix[strlen(suffix)-1] != ']')
          error->all(FLERR,"Invalid attribute in dump modify command");
        argindex[nfield+nthresh] = atoi(ptr+1);
        *ptr = '\0';
      } else argindex[nfield+nthresh] = 0;

      n = modify->find_compute(suffix);
      if (n < 0) error->all(FLERR,"Could not find dump modify compute ID");

      if (modify->compute[n]->peratom_flag == 0)
        error->all(FLERR,
                   "Dump modify compute ID does not compute per-atom info");
      if (argindex[nfield+nthresh] == 0 &&
          modify->compute[n]->size_peratom_cols > 0)
        error->all(FLERR,
                   "Dump modify compute ID does not compute per-atom vector");
      if (argindex[nfield+nthresh] > 0 &&
          modify->compute[n]->size_peratom_cols == 0)
        error->all(FLERR,
                   "Dump modify compute ID does not compute per-atom array");
      if (argindex[nfield+nthresh] > 0 &&
          argindex[nfield+nthresh] > modify->compute[n]->size_peratom_cols)
        error->all(FLERR,"Dump modify compute ID vector is not large enough");

      field2index[nfield+nthresh] = add_compute(suffix);
      delete [] suffix;

    // fix value = f_ID
    // if no trailing [], then arg is set to 0, else arg is between []
    // must grow field2index and argindex arrays, since access is beyond nfield

    } else if (strncmp(arg[1],"f_",2) == 0) {
      thresh_array[nthresh] = FIX;
      memory->grow(field2index,nfield+nthresh+1,"dump:field2index");
      memory->grow(argindex,nfield+nthresh+1,"dump:argindex");
      int n = strlen(arg[1]);
      char *suffix = new char[n];
      strcpy(suffix,&arg[1][2]);

      char *ptr = strchr(suffix,'[');
      if (ptr) {
        if (suffix[strlen(suffix)-1] != ']')
          error->all(FLERR,"Invalid attribute in dump modify command");
        argindex[nfield+nthresh] = atoi(ptr+1);
        *ptr = '\0';
      } else argindex[nfield+nthresh] = 0;

      n = modify->find_fix(suffix);
      if (n < 0) error->all(FLERR,"Could not find dump modify fix ID");

      if (modify->fix[n]->peratom_flag == 0)
        error->all(FLERR,"Dump modify fix ID does not compute per-atom info");
      if (argindex[nfield+nthresh] == 0 &&
          modify->fix[n]->size_peratom_cols > 0)
        error->all(FLERR,"Dump modify fix ID does not compute per-atom vector");
      if (argindex[nfield+nthresh] > 0 &&
          modify->fix[n]->size_peratom_cols == 0)
        error->all(FLERR,"Dump modify fix ID does not compute per-atom array");
      if (argindex[nfield+nthresh] > 0 &&
          argindex[nfield+nthresh] > modify->fix[n]->size_peratom_cols)
        error->all(FLERR,"Dump modify fix ID vector is not large enough");

      field2index[nfield+nthresh] = add_fix(suffix);
      delete [] suffix;

    // variable value = v_ID
    // must grow field2index and argindex arrays, since access is beyond nfield

    } else if (strncmp(arg[1],"v_",2) == 0) {
      thresh_array[nthresh] = VARIABLE;
      memory->grow(field2index,nfield+nthresh+1,"dump:field2index");
      memory->grow(argindex,nfield+nthresh+1,"dump:argindex");
      int n = strlen(arg[1]);
      char *suffix = new char[n];
      strcpy(suffix,&arg[1][2]);

      argindex[nfield+nthresh] = 0;

      n = input->variable->find(suffix);
      if (n < 0) error->all(FLERR,"Could not find dump modify variable name");
      if (input->variable->atomstyle(n) == 0)
        error->all(FLERR,"Dump modify variable is not atom-style variable");

      field2index[nfield+nthresh] = add_variable(suffix);
      delete [] suffix;

    } else error->all(FLERR,"Invalid dump_modify threshhold operator");

    // set operation type of threshhold

    if (strcmp(arg[2],"<") == 0) thresh_op[nthresh] = LT;
    else if (strcmp(arg[2],"<=") == 0) thresh_op[nthresh] = LE;
    else if (strcmp(arg[2],">") == 0) thresh_op[nthresh] = GT;
    else if (strcmp(arg[2],">=") == 0) thresh_op[nthresh] = GE;
    else if (strcmp(arg[2],"==") == 0) thresh_op[nthresh] = EQ;
    else if (strcmp(arg[2],"!=") == 0) thresh_op[nthresh] = NEQ;
    else error->all(FLERR,"Invalid dump_modify threshhold operator");

    // set threshhold value

    thresh_value[nthresh] = force->numeric(FLERR,arg[3]);

    nthresh++;
    return 4;
  }

  return 0;
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory in buf, choose, variable arrays
------------------------------------------------------------------------- */

bigint DumpMozi::memory_usage()
{
  bigint bytes = Dump::memory_usage();
  bytes += memory->usage(choose,maxlocal);
  bytes += memory->usage(dchoose,maxlocal);
  bytes += memory->usage(clist,maxlocal);
  bytes += memory->usage(vbuf,nvariable,maxlocal);
  return bytes;
}

/* ----------------------------------------------------------------------
   extraction of Compute, Fix, Variable results
------------------------------------------------------------------------- */

void DumpMozi::pack_compute(int n)
{
  double *vector = compute[field2index[n]]->vector_atom;
  double **array = compute[field2index[n]]->array_atom;
  int index = argindex[n];

  if (index == 0) {
    for (int i = 0; i < nchoose; i++) {
      buf[n] = vector[clist[i]];
      n += size_one;
    }
  } else {
    index--;
    for (int i = 0; i < nchoose; i++) {
      buf[n] = array[clist[i]][index];
      n += size_one;
    }
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_fix(int n)
{
  double *vector = fix[field2index[n]]->vector_atom;
  double **array = fix[field2index[n]]->array_atom;
  int index = argindex[n];

  if (index == 0) {
    for (int i = 0; i < nchoose; i++) {
      buf[n] = vector[clist[i]];
      n += size_one;
    }
  } else {
    index--;
    for (int i = 0; i < nchoose; i++) {
      buf[n] = array[clist[i]][index];
      n += size_one;
    }
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_variable(int n)
{
  double *vector = vbuf[field2index[n]];

  for (int i = 0; i < nchoose; i++) {
    buf[n] = vector[clist[i]];
    n += size_one;
  }
}

/* ----------------------------------------------------------------------
   one method for every attribute dump mozi can output
   the atom property is packed into buf starting at n with stride size_one
   customize a new attribute by adding a method
------------------------------------------------------------------------- */

void DumpMozi::pack_id(int n)
{
  int *tag = atom->tag;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = tag[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_molecule(int n)
{
  int *molecule = atom->molecule;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = molecule[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_type(int n)
{
  int *type = atom->type;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = type[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_mass(int n)
{
  int *type = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;

  if (rmass) {
    for (int i = 0; i < nchoose; i++) {
      buf[n] = rmass[clist[i]];
      n += size_one;
    }
  } else {
    for (int i = 0; i < nchoose; i++) {
      buf[n] = mass[type[clist[i]]];
      n += size_one;
    }
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_x(int n)
{
  double **x = atom->x;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = x[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_y(int n)
{
  double **x = atom->x;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = x[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_z(int n)
{
  double **x = atom->x;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = x[clist[i]][2];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_xs(int n)
{
  double **x = atom->x;

  double boxxlo = domain->boxlo[0];
  double invxprd = 1.0/domain->xprd;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = (x[clist[i]][0] - boxxlo) * invxprd;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_ys(int n)
{
  double **x = atom->x;

  double boxylo = domain->boxlo[1];
  double invyprd = 1.0/domain->yprd;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = (x[clist[i]][1] - boxylo) * invyprd;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_zs(int n)
{
  double **x = atom->x;

  double boxzlo = domain->boxlo[2];
  double invzprd = 1.0/domain->zprd;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = (x[clist[i]][2] - boxzlo) * invzprd;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_xs_triclinic(int n)
{
  int j;
  double **x = atom->x;

  double *boxlo = domain->boxlo;
  double *h_inv = domain->h_inv;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = h_inv[0]*(x[j][0]-boxlo[0]) + h_inv[5]*(x[j][1]-boxlo[1]) +
      h_inv[4]*(x[j][2]-boxlo[2]);
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_ys_triclinic(int n)
{
  int j;
  double **x = atom->x;

  double *boxlo = domain->boxlo;
  double *h_inv = domain->h_inv;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = h_inv[1]*(x[j][1]-boxlo[1]) + h_inv[3]*(x[j][2]-boxlo[2]);
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_zs_triclinic(int n)
{
  double **x = atom->x;

  double *boxlo = domain->boxlo;
  double *h_inv = domain->h_inv;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = h_inv[2]*(x[clist[i]][2]-boxlo[2]);
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_xu(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double xprd = domain->xprd;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = x[j][0] + ((image[j] & IMGMASK) - IMGMAX) * xprd;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_yu(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double yprd = domain->yprd;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = x[j][1] + ((image[j] >> IMGBITS & IMGMASK) - IMGMAX) * yprd;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_zu(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double zprd = domain->zprd;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = x[j][2] + ((image[j] >> IMG2BITS) - IMGMAX) * zprd;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_xu_triclinic(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double *h = domain->h;
  int xbox,ybox,zbox;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    xbox = (image[j] & IMGMASK) - IMGMAX;
    ybox = (image[j] >> IMGBITS & IMGMASK) - IMGMAX;
    zbox = (image[j] >> IMG2BITS) - IMGMAX;
    buf[n] = x[j][0] + h[0]*xbox + h[5]*ybox + h[4]*zbox;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_yu_triclinic(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double *h = domain->h;
  int ybox,zbox;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    ybox = (image[j] >> IMGBITS & IMGMASK) - IMGMAX;
    zbox = (image[j] >> IMG2BITS) - IMGMAX;
    buf[n] = x[j][1] + h[1]*ybox + h[3]*zbox;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_zu_triclinic(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double *h = domain->h;
  int zbox;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    zbox = (image[j] >> IMG2BITS) - IMGMAX;
    buf[n] = x[j][2] + h[2]*zbox;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_xsu(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double boxxlo = domain->boxlo[0];
  double invxprd = 1.0/domain->xprd;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = (x[j][0] - boxxlo) * invxprd + (image[j] & IMGMASK) - IMGMAX;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_ysu(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double boxylo = domain->boxlo[1];
  double invyprd = 1.0/domain->yprd;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = (x[j][1] - boxylo) * invyprd + (image[j] >> IMGBITS & IMGMASK) - IMGMAX;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_zsu(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double boxzlo = domain->boxlo[2];
  double invzprd = 1.0/domain->zprd;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = (x[j][2] - boxzlo) * invzprd + (image[j] >> IMG2BITS) - IMGMAX;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_xsu_triclinic(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double *boxlo = domain->boxlo;
  double *h_inv = domain->h_inv;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = h_inv[0]*(x[j][0]-boxlo[0]) + h_inv[5]*(x[j][1]-boxlo[1]) +
      h_inv[4]*(x[j][2]-boxlo[2]) + (image[j] & IMGMASK) - IMGMAX;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_ysu_triclinic(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double *boxlo = domain->boxlo;
  double *h_inv = domain->h_inv;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = h_inv[1]*(x[j][1]-boxlo[1]) + h_inv[3]*(x[j][2]-boxlo[2]) +
      (image[j] >> IMGBITS & IMGMASK) - IMGMAX;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_zsu_triclinic(int n)
{
  int j;
  double **x = atom->x;
  tagint *image = atom->image;

  double *boxlo = domain->boxlo;
  double *h_inv = domain->h_inv;

  for (int i = 0; i < nchoose; i++) {
    j = clist[i];
    buf[n] = h_inv[2]*(x[j][2]-boxlo[2]) + (image[j] >> IMG2BITS) - IMGMAX;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_ix(int n)
{
  tagint *image = atom->image;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = (image[clist[i]] & IMGMASK) - IMGMAX;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_iy(int n)
{
  tagint *image = atom->image;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = (image[clist[i]] >> IMGBITS & IMGMASK) - IMGMAX;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_iz(int n)
{
  tagint *image = atom->image;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = (image[clist[i]] >> IMG2BITS) - IMGMAX;
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_vx(int n)
{
  double **v = atom->v;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = v[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_vy(int n)
{
  double **v = atom->v;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = v[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_vz(int n)
{
  double **v = atom->v;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = v[clist[i]][2];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_fx(int n)
{
  double **f = atom->f;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = f[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_fy(int n)
{
  double **f = atom->f;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = f[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_fz(int n)
{
  double **f = atom->f;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = f[clist[i]][2];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_q(int n)
{
  double *q = atom->q;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = q[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_density(int n) 
{
  double *density = atom->density;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = density[clist[i]];
    n += size_one;
  }

}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_p(int n) 
{
  double *p = atom->p;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = p[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_rho(int n) 
{
  double *rho = atom->rho;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = rho[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_mux(int n)
{
  double **mu = atom->mu;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = mu[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_muy(int n)
{
  double **mu = atom->mu;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = mu[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_muz(int n)
{
  double **mu = atom->mu;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = mu[clist[i]][2];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_mu(int n)
{
  double **mu = atom->mu;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = mu[clist[i]][3];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_radius(int n)
{
  double *radius = atom->radius;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = radius[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_diameter(int n)
{
  double *radius = atom->radius;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = 2.0*radius[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_omegax(int n)
{
  double **omega = atom->omega;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = omega[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_omegay(int n)
{
  double **omega = atom->omega;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = omega[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_omegaz(int n)
{
  double **omega = atom->omega;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = omega[clist[i]][2];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_angmomx(int n)
{
  double **angmom = atom->angmom;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = angmom[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_angmomy(int n)
{
  double **angmom = atom->angmom;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = angmom[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_angmomz(int n)
{
  double **angmom = atom->angmom;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = angmom[clist[i]][2];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_tqx(int n)
{
  double **torque = atom->torque;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = torque[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_tqy(int n)
{
  double **torque = atom->torque;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = torque[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_tqz(int n)
{
  double **torque = atom->torque;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = torque[clist[i]][2];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_spin(int n)
{
  int *spin = atom->spin;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = spin[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_eradius(int n)
{
  double *eradius = atom->eradius;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = eradius[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_ervel(int n)
{
  double *ervel = atom->ervel;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = ervel[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_erforce(int n)
{
  double *erforce = atom->erforce;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = erforce[clist[i]];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_shapex(int n)
{
  double **shape = atom->shape;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = shape[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_shapey(int n)
{
  double **shape = atom->shape;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = shape[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_shapez(int n)
{
  double **shape = atom->shape;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = shape[clist[i]][2];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_quat1(int n)
{
  double **quaternion = atom->quaternion;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = quaternion[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_quat2(int n)
{
  double **quaternion = atom->quaternion;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = quaternion[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_quat3(int n)
{
  double **quaternion = atom->quaternion;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = quaternion[clist[i]][2];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_quat4(int n)
{
  double **quaternion = atom->quaternion;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = quaternion[clist[i]][3];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_blockiness1(int n)
{
  double **blockiness = atom->blockiness;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = blockiness[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_blockiness2(int n)
{
  double **blockiness = atom->blockiness;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = blockiness[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_inertiax(int n)
{
  double **inertia = atom->inertia;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = inertia[clist[i]][0];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_inertiay(int n)
{
  double **inertia = atom->inertia;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = inertia[clist[i]][1];
    n += size_one;
  }
}

/* ---------------------------------------------------------------------- */

void DumpMozi::pack_inertiaz(int n)
{
  double **inertia = atom->inertia;

  for (int i = 0; i < nchoose; i++) {
    buf[n] = inertia[clist[i]][2];
    n += size_one;
  }
}
