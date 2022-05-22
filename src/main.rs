extern crate sdl2;

use sdl2::pixels::{Color, PixelFormatEnum};
use sdl2::video::DisplayMode;
use sdl2::event::Event;
use std::time::Duration;
use sdl2::rect::Point;
use sdl2::render::{Canvas, RenderTarget};
use std::cmp;
use core::f32::{consts::PI, INFINITY};

const RESOLUTION: (u32, u32) = (800, 600);

#[derive(Debug, PartialEq)]
pub struct Object {
    vectors: Vec<Vector>,
    colors: Vec<Color>,
    // is okay here to index it like this. Object is immutable.
    faces: Vec<(usize, usize, usize)>
}

impl Object {

    pub fn cube() -> Object {
        Object {
            vectors: vec![
                Vector::from_xyz( 1.,  1., -1.),
                Vector::from_xyz( 1., -1., -1.),
                Vector::from_xyz( 1.,  1.,  1.),
                Vector::from_xyz( 1., -1.,  1.),
                Vector::from_xyz(-1.,  1., -1.),
                Vector::from_xyz(-1., -1., -1.),
                Vector::from_xyz(-1.,  1.,  1.),
                Vector::from_xyz(-1., -1.,  1.)
            ],
            colors: vec![
                Color::RGB(255, 0,   0),
                Color::RGB(0,   0,   255),
                Color::RGB(255, 0,   0),
                Color::RGB(255, 255, 0),
                Color::RGB(255, 0,   0),
                Color::RGB(255, 0,   0),
                Color::RGB(255, 0,   255),
                Color::RGB(0  , 255, 0),
            ],
            faces: vec![
                (0, 4, 2), (4, 6, 2),
                (3, 2, 7), (2, 6, 7),
                (5, 1, 7), (1, 3, 7),
                (1, 0, 3), (0, 2, 3),
                (5, 4, 1), (4, 0, 1)
            ]
        }
    }

    pub fn render<T: RenderTarget>(&self, canvas: & mut Canvas<T>, m: &Matrix) {
        let points: Vec<_> = self.vectors.iter().map(|v| m.project(&v)).collect();
        for &(i0, i1, i2) in &self.faces {
            let p0 = screen(&points[i0]);
            let p1 = screen(&points[i1]);
            let p2 = screen(&points[i2]);
            let c0 = self.colors[i0];
            let c1 = self.colors[i1];
            let c2 = self.colors[i2];
            let face = Face::new(p0, p1, p2);
            if face.orientation() {
                let (y_from, y_till) = face.height_range();
                for y in y_from..y_till {
                    let (x_from, x_till) = match face.row_intersects(y) {
                        Some(p) => p,
                        _ => continue
                    };
                    for x in x_from..x_till {

                        let p = Point::new(x, y);
                        let (u, v, w) = face.barycentric(&p);
                        let r = (c0.r as f32 * u) + (c1.r as f32* v) + (c2.r as f32 * w);
                        let g = (c0.g as f32 * u) + (c1.g as f32* v) + (c2.g as f32 * w);
                        let b = (c0.b as f32 * u) + (c1.b as f32* v) + (c2.b as f32 * w);
                        let c = Color::RGB(r as u8, g as u8, b as u8);
                        canvas.set_draw_color(c);
                        canvas.draw_point(p).unwrap(); // TODO: Horrible!
                    }
                }
                // Draw wireframe
                //canvas.set_draw_color(Color::RGB(244, 0, 0));
                //canvas.draw_line(p0, p1).unwrap();
                //canvas.draw_line(p1, p2).unwrap();
                //canvas.draw_line(p2, p0).unwrap();
            }
        }
    }
}

struct Face {
    a: Point,
    b: Point,
    c: Point
}

fn line_intersection(y: i32, p0: &Point, p1: &Point) -> Option<i32> {
    if (p0.y > y && p1.y > y) || (p0.y < y && p1.y < y) { return None }
    let p0x = p0.x as f32;
    let p1x = p1.x as f32;
    let p2x = RESOLUTION.0 as f32;
    let p3x = RESOLUTION.1 as f32;
    let p0y = p0.y as f32;
    let p1y = p1.y as f32;
    let p2y = y as f32;
    let p3y = y as f32;
    let t: f32 = 
        ((p0x-p2x)*(p2y-p3y)-(p0y-p2y)*(p2x-p3x)) /
        ((p0x-p1x)*(p2y-p3y)-(p0y-p1y)*(p2x-p3x));
    let x = (p0x + t*(p1x-p0x)).ceil();

    if x.is_normal() {
        Some(x as i32)
    } else {
        None
    }
}

impl Face {
    fn new(a: Point, b: Point, c: Point) -> Face {
        Face { a, b, c }
    }

    fn orientation(&self) -> bool {
        let e0 = (self.b.x-self.a.x)*(self.b.y+self.a.y);
        let e1 = (self.c.x-self.b.x)*(self.c.y+self.b.y);
        let e2 = (self.a.x-self.c.x)*(self.a.y+self.c.y);
        e0+e1+e2 < 0
    }

    fn row_intersects(&self, y: i32) -> Option<(i32, i32)>  {
        let (i0, i1, i2) = (
            line_intersection(y, &self.a, &self.b), 
            line_intersection(y, &self.b, &self.c), 
            line_intersection(y, &self.c, &self.a));

        match (&i0, &i1, &i2) {
            (&Some(x0), &Some(x1), &None) => (Some((cmp::min(x0, x1), cmp::max(x0, x1)))),
            (&Some(x0), &None, &Some(x1)) => (Some((cmp::min(x0, x1), cmp::max(x0, x1)))),
            (&None, &Some(x0), &Some(x1)) => (Some((cmp::min(x0, x1), cmp::max(x0, x1)))),
            // This happens in case the row exactly hits a corner. Take the lowest and biggest
            // x value from all three intersections.
            (&Some(x0), &Some(x1), &Some(x2)) => (Some((cmp::min(x0, x1), cmp::max(x1, x2)))),
            _ => None
        }
    }

    fn height_range(&self) -> (i32, i32) {
        (
            cmp::min(self.a.y, cmp::min(self.b.y, self.c.y)),
            cmp::max(self.a.y, cmp::max(self.b.y, self.c.y))
        )
    }

    fn barycentric(&self, p: &Point) -> (f32, f32, f32) {
        let vx0 = (self.b.x - self.a.x) as f32;
        let vy0 = (self.b.y - self.a.y) as f32;
        let vx1 = (self.c.x - self.a.x) as f32;
        let vy1 = (self.c.y - self.a.y) as f32;
        let vx2 = (     p.x - self.a.x) as f32;
        let vy2 = (     p.y - self.a.y) as f32;
        let den = vx0 * vy1 - vx1 * vy0;

        let v = (vx2 * vy1 - vx1 * vy2) / den;
        let w = (vx0 * vy2 - vx2 * vy0) / den;
        let u = 1. - v - w;
        (u, v, w)
    }

}

#[derive(Debug, PartialEq)]
pub struct Vector {
    x: f32,
    y: f32,
    z: f32
}

impl Vector {
    pub fn new() -> Vector {
        Vector {
            x: 0., y: 0., z: 0.
        }
    }

    pub fn from_xyz(x: f32, y: f32, z: f32) -> Vector {
        Vector {
            x, y, z
        }
    }

    pub fn dot (&self, other: &Vector) -> f32 {
        (self.x * other.x + self.y * other.y + self.z * other.z).sqrt()
    }

    pub fn cross(&self, other: &Vector) -> Vector {
        Vector {
            x: self.y*other.z - self.z*other.y,
            y: self.z*other.x - self.x*other.z,
            z: self.x*other.y - self.y*other.x 
        }
    }

    pub fn neg(&self) -> Vector {
        Vector {
            x: -self.x,
            y: -self.y,
            z: -self.z 
        }
    }

    pub fn add(&self, other: &Vector) -> Vector {
        Vector {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z
        }
    }

    pub fn min(&self, other: &Vector) -> Vector {
        Vector {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z
        }
    }

    pub fn scale(&self, scalar: f32) -> Vector {
        Vector {
            x: self.x*scalar,
            y: self.y*scalar,
            z: self.z*scalar
        }
    }

    pub fn normalize(&self) -> Vector {
        let l = (self.x*self.x + self.y*self.y + self.z*self.z).sqrt();
        let x = self.x / l;
        let y = self.y / l;
        let z = self.z / l;
        Vector {
            x: if (x+y).is_normal() { x + 0.001 } else { x },
            y: if (y+z).is_normal() { y + 0.001 } else { y },
            z: if (z+x).is_normal() { z + 0.001 } else { z }
        }

    }
}

#[derive(Debug, PartialEq)]
pub struct Matrix {
    indices: [f32; 16]
}

fn det2(m: [f32; 4]) -> f32 {
    (m[0]*m[3]) + (m[1]*m[2])
}

fn det3(m: [f32; 9]) -> f32 {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    a * det2([m[4], m[5], m[7], m[8]]) -
    b * det2([m[3], m[5], m[6], m[8]]) +
    c * det2([m[3], m[4], m[6], m[7]])
}

fn det4(m: [f32; 16]) -> f32 {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let d = m[3];
    a * det3([m[ 5], m[ 6], m[ 7], m[ 9], m[10], m[11], m[13], m[14], m[15]]) -
    b * det3([m[ 4], m[ 6], m[ 7], m[ 8], m[10], m[11], m[12], m[10], m[11]]) +
    c * det3([m[ 4], m[ 5], m[ 7], m[ 8], m[ 9], m[11], m[12], m[13], m[15]]) -
    d * det3([m[ 4], m[ 5], m[ 6], m[ 8], m[ 9], m[10], m[12], m[13], m[14]])
}

impl Matrix {
    pub fn identity() -> Matrix {
        Matrix {
            indices: [
                1., 0., 0., 0.,
                0., 1., 0., 0.,
                0., 0., 1., 0.,
                0., 0., 0., 1.]
        }
    }

    pub fn zero() -> Matrix {
        Matrix { indices: [0_f32; 16] }
    }

    pub fn look_at(eye: &Vector, target: &Vector, up: &Vector) -> Matrix {
        let z = eye.min(&target).normalize();
        let x = up.cross(&z).normalize();
        let y = z.cross(&x);
        let xw = -x.dot(&eye);
        let yw = -y.dot(&eye);
        let zw = -z.dot(&eye);
        Matrix {
            indices: [
                x.x, y.x, z.x, 0.,
                x.y, y.y, z.y, 0.,
                x.z, y.z, z.z, 0.,
                xw , yw , zw , 1.
            ]
        }
    }

   pub fn frustum(&self, r: f32, l: f32, t: f32, b: f32, n: f32, f: f32) -> Matrix {
        let x = ( 2_f32*n)   / (r-l);
        let y = ( 2_f32*n)   / (t-b);
        let z = (-2_f32*f*n) / (f-n);
        let a =  (r+l) / (r-l);
        let b =  (t+b) / (t-b);
        let c = -(f+n) / (f-n);
        let d = -1_f32;
        Matrix {
            indices: [
                x , 0., a , 0.,
                0., y , b , 0.,
                0., 0., c , z ,
                0., 0., d , 0.
            ]
        }
    }
         
    pub fn perspective(aspect: f32, y_fov: f32, z_near: f32, z_far: f32) -> Matrix {
        let f = 1_f32 / (y_fov / 2_f32).tan();
        let a = f / aspect;
        let range = 1_f32 / (z_near - z_far);
        let finite = z_far.is_finite();
        let b = if finite { (z_near + z_far) * range } else { -1. };
        let c = if finite { 2. * z_near * z_far * range } else { -2. * z_near };
        Matrix {
         indices: [
                a , 0., 0. , 0.,
                0., f , -1., 0.,
                0., 0., b  , -1.,
                0., 0., c , 0.,
            ]
        }
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        let (m1, m2) = (&other.indices, &self.indices);

        let a = m1[ 0]*m2[ 0] + m1[ 1]*m2[ 4] + m1[ 2]*m2[ 8] + m1[ 3]*m2[12];
        let e = m1[ 4]*m2[ 0] + m1[ 5]*m2[ 4] + m1[ 6]*m2[ 8] + m1[ 7]*m2[12];
        let i = m1[ 8]*m2[ 0] + m1[ 9]*m2[ 4] + m1[10]*m2[ 8] + m1[11]*m2[12];
        let m = m1[12]*m2[ 0] + m1[13]*m2[ 4] + m1[14]*m2[ 8] + m1[15]*m2[12];

        let b = m1[ 0]*m2[ 1] + m1[ 1]*m2[ 5] + m1[ 2]*m2[ 9] + m1[ 3]*m2[13];
        let f = m1[ 4]*m2[ 1] + m1[ 5]*m2[ 5] + m1[ 6]*m2[ 9] + m1[ 7]*m2[13];
        let j = m1[ 8]*m2[ 1] + m1[ 9]*m2[ 5] + m1[10]*m2[ 9] + m1[11]*m2[13];
        let n = m1[12]*m2[ 1] + m1[13]*m2[ 5] + m1[14]*m2[ 9] + m1[15]*m2[13];

        let c = m1[ 0]*m2[ 2] + m1[ 1]*m2[ 6] + m1[ 2]*m2[10] + m1[ 3]*m2[14];
        let g = m1[ 4]*m2[ 2] + m1[ 5]*m2[ 6] + m1[ 6]*m2[10] + m1[ 7]*m2[14];
        let k = m1[ 8]*m2[ 2] + m1[ 9]*m2[ 6] + m1[10]*m2[10] + m1[11]*m2[14];
        let o = m1[12]*m2[ 2] + m1[13]*m2[ 6] + m1[14]*m2[10] + m1[15]*m2[14];

        let d = m1[ 0]*m2[ 3] + m1[ 1]*m2[ 7] + m1[ 2]*m2[11] + m1[ 3]*m2[15];
        let h = m1[ 4]*m2[ 3] + m1[ 5]*m2[ 7] + m1[ 6]*m2[11] + m1[ 7]*m2[15];
        let l = m1[ 8]*m2[ 3] + m1[ 9]*m2[ 7] + m1[10]*m2[11] + m1[11]*m2[15];
        let p = m1[12]*m2[ 3] + m1[13]*m2[ 7] + m1[14]*m2[11] + m1[15]*m2[15];

        Matrix {
            indices: [
                a, b, c, d,
                e, f, g, h,
                i, j, k, l,
                m, n, o, p
            ]
        }
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        let (m1, m2) = (&self.indices, &other.indices);

        let a = m1[ 0]*m2[ 0];
        let e = m1[ 1]*m2[ 4];
        let i = m1[ 2]*m2[ 8];
        let m = m1[ 3]*m2[12];

        let b = m1[ 4]*m2[ 1];
        let f = m1[ 5]*m2[ 5];
        let j = m1[ 6]*m2[ 9];
        let n = m1[ 7]*m2[13];

        let c = m1[ 8]*m2[ 2];
        let g = m1[ 9]*m2[ 6];
        let k = m1[10]*m2[10];
        let o = m1[11]*m2[14];

        let d = m1[12]*m2[ 3];
        let h = m1[13]*m2[ 7];
        let l = m1[14]*m2[11];
        let p = m1[15]*m2[15];

        Matrix {
            indices: [
                a, b, c, d,
                e, f, g, h,
                i, j, k, l,
                m, n, o, p
            ]
        }
    }

    pub fn project(&self, v: &Vector) -> Vector {
        let m = self.indices;

        let x = m[ 0]*v.x + m[ 4]*v.y + m[ 8]*v.z + m[12];
        let y = m[ 1]*v.x + m[ 5]*v.y + m[ 9]*v.z + m[13];
        let z = m[ 2]*v.x + m[ 6]*v.y + m[10]*v.z + m[14];
        let w = m[ 3]*v.x + m[ 7]*v.y + m[11]*v.z + m[15];

        Vector {
            x: x/w, y: y/w, z: z/w // remove w component
        }
    }

    pub fn scale(&self, scalar: f32) -> Matrix {
        let m = self.indices;
        Matrix {
            indices: m.map(|x| x * scalar)
        }
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        let m1 = self.indices;
        let m2 = other.indices;
        Matrix {
            indices: [
                m1[ 0]+m2[ 0], m1[ 1]+m2[ 1], m1[ 2]+m2[ 2], m1[ 3]+m2[ 3], 
                m1[ 4]+m2[ 4], m1[ 5]+m2[ 5], m1[ 6]+m2[ 6], m1[ 7]+m2[ 7], 
                m1[ 8]+m2[ 8], m1[ 9]+m2[ 9], m1[10]+m2[10], m1[11]+m2[11], 
                m1[12]+m2[12], m1[13]+m2[13], m1[14]+m2[14], m1[15]+m2[15] 
            ]
        }
    }

    pub fn det(&self) -> f32 {
        det4(self.indices)
    }

}

fn screen(v: &Vector) -> Point {
    let x = (v.x+1.) / 2. * (RESOLUTION.0 as f32);
    let y = (v.y+1.) / 2. * (RESOLUTION.1 as f32);
    Point::new(x as i32, y as i32)
}

//fn rotate_xy(x: f32, _y: f32) -> Matrix {
//        Matrix {
//            indices: [
//            1., 0., 0., 0.,
//            0., x.cos(), -x.sin(), 0.,
//            0., x.sin(), x.cos(), 0.,
//            0., 0., 0., 1.
//            ]
//        }
//}
//
//fn _rotate(v: Vec4, x: f32, y: f32) -> Vec4 {
//    let m = <Mat4>::from_slice(&rotate_xy(x, y).indices);
//    let mut r = Matrix::<f32, 4, 1>::default();
//    r.mul_assign(m, v);
//    r
//}
//

fn main() {

    let object = Object::cube();
    let face = Face::new(Point::new(10, 2), Point::new(200, 200), Point::new(4, 200));
    dbg!(face.barycentric(&Point::new(50, 50)));
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let window = video.window("rust-sdl2 demo", RESOLUTION.0, RESOLUTION.1)
        .position_centered()
        .build()
        .unwrap();
    let mut canvas = window.into_canvas().build().unwrap();
    canvas.set_draw_color(Color::RGB(0, 0, 0));
    canvas.clear();
    let mut event_pump = sdl.event_pump().unwrap();
    let mut i = 0_f32;
    'running: loop {
        canvas.set_draw_color(Color::RGB(0, 0, 0));
        canvas.clear();
        canvas.set_draw_color(Color::RGB(244, 0, 0));
        i = i+0.1;
    
        let eye    = Vector::from_xyz(i, -6.,  0.);
        let target = Vector::from_xyz(0_f32, 0.,  0.);
        let up     = Vector::from_xyz(0_f32, 0., -1.);
        let view   = Matrix::look_at(&eye, &target, &up);
        let proj   = Matrix::perspective(1., PI/2., 5., INFINITY);

        let view_proj = &proj.dot(&view);
        object.render(& mut canvas, &view_proj);
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} | Event::KeyDown {..} => { break 'running },
                _ => {}
            }
        }

        ////
        canvas.present();        
//        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
        }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_vector_add() {
        let a = Vector::from_xyz(2., 3., 4.);
        let b = Vector::from_xyz(5., 6., 7.);
        let r = Vector::from_xyz(7., 9., 11.);
        assert_eq!(a.add(&b), r)
    }

    #[test]
    fn test_vector_min() {
        let a = Vector::from_xyz( 2.,  3.,  4.);
        let b = Vector::from_xyz( 5.,  6.,  7.);
        let r = Vector::from_xyz(-3., -3., -3.);
        assert_eq!(a.min(&b), r)
    }

    #[test]
    fn test_vector_neg() {
        let a = Vector::from_xyz(-2., 3.,  4.);
        let r = Vector::from_xyz( 2.,-3., -4.);
        assert_eq!(a.neg(), r)
    }

    #[test]
    fn test_vector_scale() {
        let a = Vector::from_xyz(-2., 3., 4.);
        let r = Vector::from_xyz(-4., 6., 8.);
        assert_eq!(a.scale(2_f32), r)
    }
    #[test]
    fn test_vector_normalize() {
        let v = Vector::from_xyz(2., 4., 2.);
        assert_eq!(v.scale(2_f32).normalize(), v.normalize())
    }

    #[test]
    fn test_vector_cross() {
        let a = Vector::from_xyz( 2., 3.,  4.);
        let b = Vector::from_xyz( 5., 6.,  7.);
        let r = Vector::from_xyz(-3., 6., -3.);
        assert_eq!(a.cross(&b), r)
    }

    #[test]
    fn test_vector_dot() {
        let a = Vector::from_xyz(4., 8., 10.);
        let b = Vector::from_xyz(9., 2., 7.);
        assert_eq!(a.dot(&b), 122_f32.sqrt())
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity();
        assert_eq!(m.indices, [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.]); 
    }

    #[test]
    fn test_matrix_det() {
        let m1 = Matrix::identity();
        let mut m2 = Matrix::identity();
        m2.indices[0] = 2_f32;
        assert_eq!(m1.det(), 1_f32);
        assert_eq!(m2.det(), 2_f32);
    }

    #[test]
    fn test_matrix_dot() {
        let mut a = Matrix::zero();
        a.indices = [
            1., 2., 3., 4.,
            4., 3., 2., 1.,
            1., 1., 1., 2.,
            2., 2., 3., 1.];
        let mut b = Matrix::zero();
        b.indices = [
            8., 2., 3., 0.,
            4., 3., 2., 1.,
            1., 7., 1., 8.,
            2., 2., 2., 1.];
        let mut c = Matrix::zero();
        c.indices = [
            1., 2., 2., 0.,
            0., 1., 4., 1.,
            1., 1., 1., 3.,
            3., 3., 1., 1.];
        let i = Matrix::identity();
        let o = Matrix::zero();
        assert_eq!(a.dot(&b).dot(&c), a.dot(&b.dot(&c)));
        assert_ne!(a.dot(&b), b.dot(&a));
        assert_eq!(a.dot(&b.add(&c)), a.dot(&b).add(&a.dot(&c)));
        assert_eq!(i.dot(&a), a);
        assert_eq!(a.dot(&i), a);
        assert_eq!(o.dot(&a), o);
        assert_eq!(a.dot(&o), o);
        //let m1 = Matrix::identity();
        //let m2 = Matrix::identity();
        //let m3 = Matrix::zero();
        //assert_eq!(m1.dot(&m2), m1);
        //assert_eq!(m1.dot(&m3), m3);
    }

    #[test]
    fn test_matrix_scale() {
        let m = Matrix::identity().scale(2_f32);
        assert_eq!(m.indices, [2., 0., 0., 0., 0., 2., 0., 0., 0., 0., 2., 0., 0., 0., 0., 2.]); 
    }

}
