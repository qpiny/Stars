package org.rejna.stars;

import com.jogamp.opencl.*;
import com.jogamp.opencl.CLMemory.Mem.*;

public class NCorps {

	  public static final int N = 1024;
	  public static final int ITER = 10000;
	  public static final int THREAD_N = 4;
	  public static final int BLOCK = N/THREAD_N;
	  public static final float DT = 0.1f;
	  public static final float G = 0.1f;
	  public static final float EPSILON = 1;

	  static final String programSource =
	"__kernel void euler("+
	"                    __global float *sh_x"+
	"                  , __global const float *x"+
	"                  , __global float *dx"+
	"                  , __global float *d2x"+
	"                  , __global float *sh_y"+
	"                  , __global const float *y"+
	"                  , __global float *dy"+
	"                  , __global float *d2y"+
	"                  , __global float *sh_z"+
	"                  , __global const float *z"+
	"                  , __global float *dz"+
	"                  , __global float *d2z"+
	"                  , __global const float *mass"+
	"                ){"+
	        "    int gid = get_global_id(0);"+
	        "    int lid = get_local_id(0);"+
	        "    int gs = get_global_size(0);"+
	        "    int ls = get_local_size(0);"+
	        "    int nbb = gs/ls;"+
	/*"    //__local float lx[256];"+
		"    //__local float ly[256];"+
		"    //__local float lz[256];"+
		"    //__local float lm[256];"+*/
	        "    float ax = 0;"+
	        "    float ay = 0;"+
	        "    float az = 0;"+
	        "    const float x0 = x[gid];"+
	        "    const float y0 = y[gid];"+
	        "    const float z0 = z[gid];"+
	        /*"    //for(int j = 0; j < nbb; j++){"+
		"    //  lx[lid] = x[j*ls+lid];"+
		"    //  ly[lid] = y[j*ls+lid];"+
		"    //  lz[lid] = z[j*ls+lid];"+
		"    //  lm[lid] = mass[j*ls+lid];"+
		"    //  barrier(CLK_LOCAL_MEM_FENCE);"+
		"      //for(int i = 0; i < ls;i++){"+*/
	"      for(int i = 0; i < "+N+";i++){"+
	/*"        //if(j*ls+i == gid){ continue; }"+*/
	"        if(i == gid){ continue; }"+
	/*"        //float rx = x0-lx[i];"+
		"        //float ry = y0-ly[i];"+
		"        //float rz = z0-lz[i];"+*/
	"        float rx = x0-x[i];"+
	"        float ry = y0-y[i];"+
	"        float rz = z0-z[i];"+
	"        float r2 = rx*rx+ry*ry+rz*rz+"+EPSILON+";"+
	"        float r = sqrt(r2);"+
	/*"        //float df = "+G+"*lm[i]/r2/r;"+*/
	"        float df = "+G+"*mass[i]/r2/r;"+
	"        ax -= df*rx;"+
	"        ay -= df*ry;"+
	"        az -= df*rz;"+
	        "      }"+
	/*"    //  barrier(CLK_LOCAL_MEM_FENCE);"+
	        "    //}"+
	        "    //}"+*/
	"    d2x[gid] = ax;"+
	"    d2y[gid] = ay;"+
	"    d2z[gid] = az;"+
	        "    sh_x[gid] = x0+"+DT+"*dx[gid] + 0.5*"+DT+"*"+DT+"*d2x[gid];"+
	        "    sh_y[gid] = y0+"+DT+"*dy[gid] + 0.5*"+DT+"*"+DT+"*d2y[gid];"+
	        "    sh_z[gid] = z0+"+DT+"*dz[gid] + 0.5*"+DT+"*"+DT+"*d2z[gid];"+
	        "    dx[gid] += "+DT+"*d2x[gid];"+
	        "    dy[gid] += "+DT+"*d2y[gid];"+
	        "    dz[gid] += "+DT+"*d2z[gid];"+
	        "}"
	;

	  class OCLConfig{
	    int platformIndex = 0;
	    int deviceIndex = 0;
	  };
	  OCLConfig config = new OCLConfig();
	  
	  private void buildConfigFromArgs(String[] args){
	    for(int i = 0; i < args.length; i++){
	      System.out.printf("analysing %s\n", args[i]);
	      if(args[i].equals("-dev") && (args.length > i+1)){
	        i++;
	        config.deviceIndex = Integer.parseInt(args[i]);
	      }
	      else{
	        System.out.printf("usage : java [class] -dev n\n");
	      }
	    }
	  }

	  public static void main(final String[] args){
	    NCorps exec = new NCorps();
	    exec.go(args);
	  }

	  static float[] x = new float[N];
	  static float[] y = new float[N];
	  static float[] z = new float[N];
	  static float[] shadow_x = new float[N];
	  static float[] shadow_y = new float[N];
	  static float[] shadow_z = new float[N];
	  static float[] dx = new float[N];
	  static float[] dy = new float[N];
	  static float[] dz = new float[N];
	  static float[] shadow_dx = new float[N];
	  static float[] shadow_dy = new float[N];
	  static float[] shadow_dz = new float[N];
	  static float[] d2x = new float[N];
	  static float[] d2y = new float[N];
	  static float[] d2z = new float[N];
	  static float[] shadow_d2x = new float[N];
	  static float[] shadow_d2y = new float[N];
	  static float[] shadow_d2z = new float[N];
	  static float[] mass = new float[N];
	  class Meteor
	  {
	    final int idx;

	    Meteor(final int index){
	      this.idx = index;
	    }

	    public void setInitPostion(final float px, final float py, final float pz){
	      x[this.idx] = px;
	      y[this.idx] = py;
	      z[this.idx] = pz;
	    }
	    public void setInitSpeed(final float sx, final float sy, final float sz){
	      dx[this.idx] = sx;
	      dy[this.idx] = sy;
	      dz[this.idx] = sz;
	    }
	    public void eulerIt(){
	      shadow_x[this.idx] += DT*dx[this.idx];
	      shadow_y[this.idx] += DT*dy[this.idx];
	      shadow_z[this.idx] += DT*dz[this.idx];
	      shadow_dx[this.idx] += DT*d2x[this.idx];
	      shadow_dy[this.idx] += DT*d2y[this.idx];
	      shadow_dz[this.idx] += DT*d2z[this.idx];
	      shadow_d2x[this.idx] += 0;
	      shadow_d2y[this.idx] += 0;
	      shadow_d2z[this.idx] += 0;
	      for(int i = 0 ; i < NCorps.this.meteors.length; i++){
	final Meteor tmp = NCorps.this.meteors[i];
	if(this == tmp){
	  continue;
	}
	final float rx = x[this.idx] - x[tmp.idx]
	          , ry = y[this.idx] - y[tmp.idx]
	          , rz = z[this.idx] - z[tmp.idx];
	        final float r2 = rx*rx+ry*ry+rz*rz;
	        final float r = (float)Math.sqrt(r2);
	float df = G*mass[tmp.idx]/r2/r;
	shadow_d2x[this.idx] += df*rx;
	shadow_d2y[this.idx] += df*ry;
	shadow_d2z[this.idx] += df*rz;
	      }
	    }
	  };
	  final Meteor[] meteors = new Meteor[N];
	  void swap(){
	    float[] tmp = x;
	    x = shadow_x;
	    shadow_x = tmp;
	    tmp = y;
	    y = shadow_y;
	    shadow_y = tmp;
	    tmp = z;
	    z = shadow_z;
	    shadow_z = tmp;
	    tmp = dx;
	    dx = shadow_dx;
	    shadow_dx = tmp;
	    tmp = dy;
	    dy = shadow_dy;
	    shadow_dy = tmp;
	    tmp = dz;
	    dz = shadow_dz;
	    shadow_dz = tmp;
	    tmp = d2x;
	    d2x = shadow_d2x;
	    shadow_d2x = tmp;
	    tmp = d2y;
	    d2y = shadow_d2y;
	    shadow_d2y = tmp;
	    tmp = d2z;
	    d2z = shadow_d2z;
	    shadow_d2z = tmp;
	  }

	  public void go(final String[] args){
	    buildConfigFromArgs(args);
	    CLContext context = CLContext.create();

	    // always make sure to release the context under all circumstances
	    // not needed for this particular sample but recommented
	    try {

	      // select fastest device
	      CLDevice device = context.getMaxFlopsDevice();

	      // create command queue on device.
	      CLCommandQueue queue = device.createCommandQueue();
	      
	      CLProgram program = context.createProgram(programSource).build();
	      
	      CLKernel kernelEuler = program.createCLKernel("VectorAdd");
	    //kernelRFD = clCreateKernel(program, "rfd", null);
	    final long start = System.nanoTime();
	    long time = System.nanoTime()-start;
	    for(int i = 0 ; i < this.meteors.length; i++){
	      this.meteors[i] = new Meteor(i);
	      this.meteors[i].setInitPostion((float)(Math.random()*100-50), (float)(Math.random()*100-50), (float)(Math.random()*100-50));
	      this.meteors[i].setInitSpeed((float)(Math.random()*100-50), (float)(Math.random()*100-50), (float)(Math.random()*100-50));
	      mass[i] = 1;
	    }
	    context.createBuffer(null, arg1)
//	    final Pointer px = Pointer.to(x);
//	    final Pointer py = Pointer.to(y);
//	    final Pointer pz = Pointer.to(z);
//	    final Pointer pdx = Pointer.to(dx);
//	    final Pointer pdy = Pointer.to(dy);
//	    final Pointer pdz = Pointer.to(dz);
//	    final Pointer pd2x = Pointer.to(d2x);
//	    final Pointer pd2y = Pointer.to(d2y);
//	    final Pointer pd2z = Pointer.to(d2z);
//	    final Pointer pshadow_x = Pointer.to(shadow_x);
//	    final Pointer pshadow_y = Pointer.to(shadow_y);
//	    final Pointer pshadow_z = Pointer.to(shadow_z);
//	    final Pointer pshadow_dx = Pointer.to(shadow_dx);
//	    final Pointer pshadow_dy = Pointer.to(shadow_dy);
//	    final Pointer pshadow_dz = Pointer.to(shadow_dz);
//	    final Pointer pshadow_d2x = Pointer.to(shadow_d2x);
//	    final Pointer pshadow_d2y = Pointer.to(shadow_d2y);
//	    final Pointer pshadow_d2z = Pointer.to(shadow_d2z);
//	    final Pointer pmass = Pointer.to(mass);
	    cl_mem srcMem_x = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_y = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_z = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_shadow_x = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_shadow_y = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_shadow_z = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_dx = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_dy = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_dz = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_shadow_dx = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_shadow_dy = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_shadow_dz = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_d2x = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_d2y = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_d2z = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_shadow_d2x = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_shadow_d2y = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_shadow_d2z = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_float * N, null, null);
	    cl_mem srcMem_mass = clCreateBuffer(context, CL_MEM_READ_ONLY, Sizeof.cl_float * N, null, null);
	    // Enqueueing buffers
	    clEnqueueWriteBuffer(commandQueue, srcMem_x, CL_TRUE, 0, N * Sizeof.cl_float, px, 0, null, null);
	    clEnqueueWriteBuffer(commandQueue, srcMem_y, CL_TRUE, 0, N * Sizeof.cl_float, py, 0, null, null);
	    clEnqueueWriteBuffer(commandQueue, srcMem_z, CL_TRUE, 0, N * Sizeof.cl_float, pz, 0, null, null);
	    clEnqueueWriteBuffer(commandQueue, srcMem_shadow_dx, CL_TRUE, 0, N * Sizeof.cl_float, pshadow_dx, 0, null, null);
	    clEnqueueWriteBuffer(commandQueue, srcMem_shadow_dy, CL_TRUE, 0, N * Sizeof.cl_float, pshadow_dy, 0, null, null);
	    clEnqueueWriteBuffer(commandQueue, srcMem_shadow_dz, CL_TRUE, 0, N * Sizeof.cl_float, pshadow_dz, 0, null, null);
	    clEnqueueWriteBuffer(commandQueue, srcMem_mass, CL_TRUE, 0, N * Sizeof.cl_float, pmass, 0, null, null);
	    int argn=0;
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_shadow_x));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_x));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_shadow_dx));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_shadow_d2x));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_shadow_y));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_y));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_shadow_dy));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_shadow_d2y));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_shadow_z));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_z));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_shadow_dz));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_shadow_d2z));
	    clSetKernelArg(kernelEuler, argn++, Sizeof.cl_mem, Pointer.to(srcMem_mass));
	    System.err.printf("initialisation : %ds %03d %03d %03d\n", (int)(time/1e9), ((int)(time/1e6)%1000), ((int)(time/1e3)%1000), time%1000);
	    final long exec = System.nanoTime();
	    final long[] gs = new long[]{N};
	    final long[] ls = null;//(0 == config.deviceIndex)?null:new long[]{256};
	    for(int n = 0 ; n < ITER/2; n++){
	      clSetKernelArg(kernelEuler, 0, Sizeof.cl_mem, Pointer.to(srcMem_shadow_x));
	      clSetKernelArg(kernelEuler, 1, Sizeof.cl_mem, Pointer.to(srcMem_x));
	      clSetKernelArg(kernelEuler, 4, Sizeof.cl_mem, Pointer.to(srcMem_shadow_y));
	      clSetKernelArg(kernelEuler, 5, Sizeof.cl_mem, Pointer.to(srcMem_y));
	      clSetKernelArg(kernelEuler, 8, Sizeof.cl_mem, Pointer.to(srcMem_shadow_z));
	      clSetKernelArg(kernelEuler, 9, Sizeof.cl_mem, Pointer.to(srcMem_z));
	      clEnqueueNDRangeKernel(commandQueue, kernelEuler, 1, null,
	            gs, ls, 0, null, null);
	      //clFinish(commandQueue);
	      clSetKernelArg(kernelEuler, 1, Sizeof.cl_mem, Pointer.to(srcMem_shadow_x));
	      clSetKernelArg(kernelEuler, 0, Sizeof.cl_mem, Pointer.to(srcMem_x));
	      clSetKernelArg(kernelEuler, 5, Sizeof.cl_mem, Pointer.to(srcMem_shadow_y));
	      clSetKernelArg(kernelEuler, 4, Sizeof.cl_mem, Pointer.to(srcMem_y));
	      clSetKernelArg(kernelEuler, 9, Sizeof.cl_mem, Pointer.to(srcMem_shadow_z));
	      clSetKernelArg(kernelEuler, 8, Sizeof.cl_mem, Pointer.to(srcMem_z));
	      clEnqueueNDRangeKernel(commandQueue, kernelEuler, 1, null,
	            gs, ls, 0, null, null);
	      //clFinish(commandQueue);
	    }
	    clFinish(commandQueue);
	    time = (System.nanoTime()-exec)/ITER;
	    System.err.printf("execution : %ds %03d %03d %03d\n", (int)(time/1e9), ((int)(time/1e6)%1000), ((int)(time/1e3)%1000), time%1000);
	    time = System.nanoTime()-start;
	    System.err.printf("total : %ds %03d %03d %03d\n", (int)(time/1e9), ((int)(time/1e6)%1000), ((int)(time/1e3)%1000), time%1000);
	  }
	  private static String getDeviceString(cl_device_id device, int paramName){
	      long size[] = new long[1];
	      clGetDeviceInfo(device, paramName, 0, null, size);
	      byte buffer[] = new byte[(int)size[0]];
	      clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);
	      return new String(buffer, 0, buffer.length-1);
	    }
	}

	          