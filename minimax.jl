using Random, Distributions
using ForwardDiff, LinearAlgebra

Random.seed!(666)

# Minimax Object

struct minimax
    
    x
    y 
    z
    alpha
    T
    
end

# Auxiliary functions

concatenate(x) = [v[1]  for v in x]


function minimax_solver(L, method, nx = 1, ny = 1, T = 100, alpha = 0.1, initialization = false, gfactor = 1, mu0 = 1, mumax = 1e6)
    
    #   Implementation of the twisted gradient descent for minimaximization problems
    #
    #   Arguments:
    #       . L         function. function to optimize
    #       . method    GD, OMD, ITD
    #       . nx        integer. The first nx coordinates of the argument of L correspond to the 
    #                   variable x with respect we minimize
    #       . ny        integer. From the nx+1 to nx+ny following indeces correspond to y
    #       . T         integer. Total number of steps
    #       . alpha     float. stepsize
    #       . initialization 
    
    
    # Differentiation 
    gradient = x -> ForwardDiff.gradient(L, x)

    if method in ["ITD", "IOD", "227","adaptive ITD",  "adaptive 227", "adaptive IOD"]
    
        hessian  = x -> ForwardDiff.hessian(L, x)
        I = Diagonal( ones(nx + ny) ) 
        J = Diagonal( vcat( ones(nx), - ones(ny) ) )    
    
    end

    
    # Initialization
    
    if isa(initialization, Array)
        @assert nx + ny == length(initialization)
        z0 = convert(Array{Float64}, initialization)
    else
        z0 = ones(nx + ny)
    end
    x0 = z0[1:nx]
    y0 = z0[(nx + 1): (nx + ny)]
    
    
    x = [ x0 ]
    y = [ y0 ]
    z = [ z0 ]
    
    ### Gradient Descent (GD)

    if method == "GD"

        stepsize = alpha

        for t = 1:(T-1)
            
            grad = gradient(vcat( x[t], y[t] ))

            grad_x = grad[1:nx]
            grad_y = grad[(nx+1): nx + ny]
            
            push!(x, x[t] - alpha * grad_x)
            push!(y, y[t] + alpha * grad_y)
            push!(z, vcat(x[t+1], y[t+1]))
            
        end

    end

    ### Optimistic Mirror Descent (OMD)

    if method == "OMD"

        stepsize = alpha

        global grad_x = "empty array"
        global grad_y = "empty array"

        for t = 1:(T-1)
        
            if t > 1
                old_grad_x = grad_x
                old_grad_y = grad_y
            end
            
            grad = gradient(vcat( x[t], y[t] ))
    
            global grad_x = grad[1:nx]
            global grad_y = grad[(nx+1): nx + ny]
            
            if t == 1
    
                push!(x, x[t] - alpha * grad_x)
                push!(y, y[t] + alpha * grad_y)
                push!(z, vcat(x[t+1], y[t+1]))
                continue
                
            end
            
            push!(x, x[t] - 2 * alpha * grad_x + alpha * old_grad_x)
            push!(y, y[t] + 2 * alpha * grad_y - alpha * old_grad_y)
            push!(z, vcat(x[t+1], y[t+1]))

        end  

    end

    ### Implicit Twisted Descent (ITD)

    if method == "ITD"

        stepsize = alpha

        for t = 1:(T-1)
            
            grad = gradient(z[t])
            hess = hessian(z[t])

            shift = inv( J + gfactor * alpha * hess ) * grad
            
            push!(z, z[t] - alpha * shift)
            #push!(z, z[t] - gfactor * alpha * shift)
                    
            push!(x, z[t+1][1:nx])
            push!(y, z[t+1][(nx+1):(nx+ny)])        
        
            
        end

    end

    ### Facu 227
 
    if method == "227"

        stepsize = alpha

        for t = 1:(T-1)
            
            grad = gradient(z[t])
            hess = hessian(z[t])

            grad_x = grad[1:nx]
            grad_y = grad[(nx+1):(nx + ny)]

            Hxy = hess[1:nx, (nx + 1):(nx + ny)]
            Hyx = hess[(nx+1) : (nx + ny), 1 : nx]

            if nx == 1 && ny == 1 
                shift_x = alpha * ( grad_x[1] + alpha * Hxy[1] * grad_y[1] ) / ( 1 + alpha^2 * Hxy[1] * Hyx[1] )
                shift_y = alpha * ( grad_y[1] - alpha * Hyx[1] * grad_x[1] ) / ( 1 + alpha^2 * Hyx[1] * Hxy[1] )
            else
                shift_x = alpha * inv( Diagonal(ones(nx)) + alpha^2 * Hxy * Hyx ) * ( grad_x + alpha * Hxy * grad_y )
                shift_y = alpha * inv( Diagonal(ones(ny)) + alpha^2 * Hyx * Hxy ) * ( grad_y - alpha * Hyx * grad_x )
            end

            push!(x, x[t] .- shift_x)
            push!(y, y[t] .+ shift_y)
            push!(z, vcat(x[t+1], y[t+1]))
            
        end

    end

    ### Implicit Twisted Descent (ITD) with adaptive stepsize

    if method == "adaptive ITD"

        @assert alpha > 1

        mu = mu0
        stepsize = []

        for t = 1:(T-1)

            grad = gradient(z[t])
            hess = hessian(z[t])      

            if norm(grad) < 1e-30
                break
            end

            mu = min( alpha * mu, mumax)
            eta = mu / ( norm(grad)^2 )

            satisfied_condition = false

            while !satisfied_condition

                shift = inv( J + eta * hess ) * grad
    
                global new_z = z[t] - eta * shift 
                global new_x = new_z[1:nx]
                global new_y = new_z[(nx+1 : (nx + ny))]
                
                if L( vcat(new_x, y[t]) ) <= L( vcat(new_x, new_y) ) <= L( vcat(x[t], new_y) )
                    satisfied_condition = true
                elseif eta < 1e-20
                    satisfied_condition = true
                else
                    eta = eta / 2
                end

            end

            push!(z, new_z)
            push!(x, new_x)
            push!(y, new_y)        
            push!(stepsize, eta)

        end

    end

    ### Adaptive 227

    if method == "adaptive 227"

        @assert alpha > 1

        mu = mu0
        stepsize = []

        for t = 1:(T-1)

            grad = gradient(z[t])
            if norm(grad) < 1e-30
                break
                println("Stop because of vanishing gradient")
            end
            hess = hessian(z[t])        
            
            grad_x = grad[1:nx]
            grad_y = grad[(nx+1):(nx + ny)]

            Hxy = hess[1:nx, (nx + 1):(nx + ny)]
            Hyx = hess[(nx+1) : (nx + ny), 1 : nx]
        

            mu = min( alpha * mu, mumax)
            eta = mu #/ ( norm(grad)^2 )

            satisfied_condition = false

            while !satisfied_condition

                if nx == 1 && ny == 1 
                    shift_x = eta * ( grad_x[1] + eta * Hxy[1] * grad_y[1] ) / ( 1 + eta^2 * Hxy[1] * Hyx[1] )
                    shift_y = eta * ( grad_y[1] - eta * Hyx[1] * grad_x[1] ) / ( 1 + eta^2 * Hyx[1] * Hxy[1] )
                else
                    shift_x = eta * inv( Diagonal(ones(nx)) + eta^2 * Hxy * Hyx ) * ( grad_x + eta * Hxy * grad_y )
                    shift_y = eta * inv( Diagonal(ones(ny)) + eta^2 * Hyx * Hxy ) * ( grad_y - eta * Hyx * grad_x )
                end
    
                global new_x = x[t] .- shift_x
                global new_y = y[t] .+ shift_y
                
                if L( vcat(new_x, y[t]) ) <= L( vcat(new_x, new_y) ) <= L( vcat(x[t], new_y) )
                    satisfied_condition = true
                elseif eta < 1e-30
                    satisfied_condition = true
                else
                    eta = eta / 2
                end

            end

            push!(z, vcat(new_x, new_y))
            push!(x, new_x)
            push!(y, new_y)        
            push!(stepsize, eta)
            
        end

    end

    ### Implicit Optimistic Descent (IOD)

    if method == "IOD"

        stepsize = alpha
        global grad = "empty array"

        for t = 1:(T-1)
        
            if t > 1
                old_grad = grad
            end
            
            grad = gradient(z[t])
            hess = hessian(z[t])

            inverse = inv( J + gfactor * alpha * hess ) 

            if t == 1
    
                push!(z , z[t] - alpha * inverse * grad)

                push!(x, z[t+1][1 : nx])
                push!(y, z[t+1][(nx + 1) : (nx + ny)])
                continue
                
            end
            
            push!(z, z[t] - alpha * inverse * ( 2 * grad - old_grad) )
            push!(x, z[t+1][1 : nx])
            push!(y, z[t+1][(nx + 1) : (nx + ny)])

        end  

    end

    ### adapive Implicit Optimistic Descent (IOD)

    if method == "adaptive IOD"

        @assert alpha > 1
        mu = mu0
        global grad = "empty array"
        stepsize = []

        for t = 1:(T-1)
            
            if t > 1
                old_grad = grad
            end

            grad = gradient(z[t])
            #hess = hessian(z[t])

            if norm(grad) < 1e-30
                break
            end

            mu = min( alpha * mu, mumax)
            eta = mu / ( norm(grad)^2 )

            if t == 1
    
                #push!(z , z[t] - eta * inv( J + gfactor * eta * hess ) * grad)
                push!(z , z[t] - eta * grad)
                push!(x, z[t+1][1 : nx])
                push!(y, z[t+1][(nx + 1) : (nx + ny)])
                continue
                
            end
            
            satisfied_condition = false

            while !satisfied_condition

                shift = eta * ( 2 * grad - old_grad) 
                #shift = eta * inv( J + gfactor * eta * hess ) * ( 2 * grad - old_grad) 
    
                global new_z = z[t] - shift 
                global new_x = new_z[1:nx]
                global new_y = new_z[(nx+1 : (nx + ny))]
                
                if L( vcat(new_x, y[t]) ) <= L( vcat(new_x, new_y) ) <= L( vcat(x[t], new_y) )
                    satisfied_condition = true
                elseif eta < 1e-20
                    satisfied_condition = true
                else
                    eta = eta / 2
                end

            end

            push!(z, new_z )
            push!(x, new_x )
            push!(y, new_y )
            push!(stepsize, eta)

        end  

    end

    ################

    # Returns

    if nx == 1 && ny == 1
        x = concatenate(x)
        y = concatenate(y)
    end


    return minimax(x,
                   y,
                   z,
                   stepsize,
                   T)
    
end