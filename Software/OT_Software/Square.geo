// gmsh cube.geo -3 -setnumber lc 1.0 -setnumber numLayers 20 -o cube_mesh.msh -nopopup
// Define default values for variables
// numLayers = 40; // Number of extrusion layers

// Define points for the square in the xy-plane
Point(1) = {-1, -1, 0};
Point(2) = {1, -1, 0};
Point(3) = {1, 1, 0};
Point(4) = {-1, 1, 0};

// Define lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Define line loop and surface
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Specify transfinite lines and surface to ensure quadrilateral elements
Transfinite Line {1, 2, 3, 4} = numLayers+1 Using Progression 1;
Transfinite Surface {1};
Recombine Surface {1};

// Mesh the surface
Mesh 2;


// Physical groups
Physical Point("1") = {1};
Physical Point("2") = {2};
Physical Point("3") = {3};
Physical Point("4") = {4};
Physical Curve("Abajo") = {1};
Physical Curve("Derecha") = {2};
Physical Curve("Arriba") = {3};
Physical Curve("Izquierda") = {4};
Physical Surface("Volumen") = {1};

// Save the mesh to a file
Save "Square.msh";