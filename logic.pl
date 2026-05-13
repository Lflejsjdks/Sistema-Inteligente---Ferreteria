% Reglas lógicas para Distribuciones Molina
necesita_reabastecer(Producto) :- 
    stock_actual(Producto, Cantidad), 
    stock_minimo(Producto, Minimo), 
    Cantidad < Minimo.