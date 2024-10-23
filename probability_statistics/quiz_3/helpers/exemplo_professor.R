#####################################################################################
#                                                                                   #
#                                      N O R M A L                                  #
#                                                                                   #
#####################################################################################

#*********************       INTEGRAL         #*********************

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


gamma(8)
factorial(7)
7*6*5*4*3*2*1


# a) Z < 1.25
mi <- 0;
sigma <- 1
phi <- function(x) (1/(sigma*sqrt(2*pi)))*exp((-1/2)*((x-mi)/sigma)^2)
integrate(phi,-Inf,1.25)
plot(phi,-5,5,lwd=3)

polygon(x=c(-5,seq(-5,1.25,l=100),1.25), y=c(0,phi(seq(-5,1.25,l=100)), 0), col="gray")


# b) Z > 1.25
mi <- 0;
sigma <- 1
phi <- function(x) (1/(sigma*sqrt(2*pi)))*exp((-1/2)*((x-mi)/sigma)^2)
I <- integrate(phi,-Inf,1.25)

1-integrate(phi,-Inf,1.25)$value
1-I$value

# c) Z < -1.25
mi <- 0;
sigma <- 1
phi <- function(x) (1/(sigma*sqrt(2*pi)))*exp((-1/2)*((x-mi)/sigma)^2)
integrate(phi,-Inf,-1.25)

# d) -0.38 < Z < 1.25
mi <- 0;
sigma <- 1
phi <- function(x) (1/(sigma*sqrt(2*pi)))*exp((-1/2)*((x-mi)/sigma)^2)
integrate(phi,-0.38,1.25)



#*********************       PNORM         #*********************

# a) Z < 1.25
pnorm(1.25,mean=0,sd=1)         # fun??o de probabilidade acumulada da normal
pnorm(1.25,0,1)
pnorm(1.25)

# b) Z > 1.25
1-pnorm(1.25,0,1) 

# c) Z < -1.25
pnorm(-1.25,0,1)

# d) -0.38 < Z < 1.25
pnorm(1.25,0,1)-pnorm(-0.38,0,1) 


#*********************       QNORM         #*********************

# a) alpha = 0.95
qnorm(0.95)                 #   fornece o quantil q para uma probabilidade p (?rea de -??? at? q)

# b) alpha = 0.975
qnorm(0.975)

# c) alpha = 0.995
qnorm(0.995)

# d) alpha = 0.95; alpha = 0.975; alpha = 0.995
qnorm(c(0.95,0.975,0.995))



######## PLOTAR A CURVA NORMAL #########
dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

a <- seq(4, 6, length=1000)                  # PLOTA UMA CURVA 
b <- dnorm(a, mean=5, sd=0.1)
plot(a, b, type="l", lwd=5, col="blue")


x <- seq(4, 6, length=1000)                 # PLOTA OUTRA CURVA
y <- dnorm(x, mean=5, sd=0.2)
plot(x, y, type="l", lwd=5, col="red")

#____________________________
x <- seq(4, 6, length=1000)                 # PLOTA AS DUAS CURVAS
y1 <- dnorm(x, mean=5, sd=0.1)
y2 <- dnorm(x, mean=5, sd=0.2)
plot(x, y1, type="l", lwd=5, col="blue")
lines(x,y2, lwd=5,col="red")


#####################################################################################
#                                                                                   #
#                               Q U I - Q U A D R A D O                             #
#                                                                                   #
#####################################################################################

#*********************       INTEGRAL         #*********************

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


# FUN??O DENSIDADE DE PROBABILIDADE
df <- 10;
chisq <- function(x) (1/((2^(df/2))*gamma(df/2)))*(x^((df/2)-1))*exp(-x/2)
integrate(chisq,10,Inf)
plot(chisq,0,15,lwd=3, col="blue")

# PLOTAR A FUN??O DE DISTRUBUI??O PARA ALGUNS GRAUS DE LIBERDADE
curve(dchisq(x, df = 3), from = 0, to = 20, ylab = "y", lwd=3)
dfs <- c(4, 5, 10, 15)
for (i in dfs) curve(dchisq(x, df = i), 0, 20, lwd=3, add = TRUE)


#*********************       EXERCICIO 5         #*********************

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

pchisq(0.484,4, lower.tail = FALSE)
pchisq(11.14,4, lower.tail = FALSE)

df <- 4;
chisq <- function(x) (1/((2^(df/2))*gamma(df/2)))*(x^((df/2)-1))*exp(-x/2)
plot(chisq,0,15,lwd=5, col="blue")
polygon(x=c(0.484,seq(0.484,11.143,l=100),11.143), y=c(0,chisq(seq(0.484,11.143,l=100)), 0), col="gray")
abline(h=0,lwd=3,col="black")
abline(v=c(0.484,3.26,11.143), col=c("black","red", "black"), lty=c(1,2,1), lwd=c(1,3,1))
text(c(0.95,3.8, 10.3), c(0.009,0.009, 0.009), c(expression(0.484,3.26,11.143)), 
     col=c("black","red", "black")) 



#*********************       PCHISQ         #*********************

pchisq()   # Usado para a fun??o de densidade acumulada

pchisq(18.30704,10, lower.tail = FALSE)

pchisq(18.30704,10, lower.tail = TRUE)



#*********************       QCHISQ         #*********************

qchisq()   # fornece o quantil q para uma probabilidade p


qchisq(0.05,10, lower.tail = FALSE)

qchisq(0.95,3, lower.tail = TRUE)


# a) 
qchisq(0.1,15, lower.tail = FALSE)
round(qchisq(0.1,15, lower.tail = FALSE),3)

# b) 
qchisq(0.005,25, lower.tail = FALSE)

# c) 
qchisq(0.99,25, lower.tail = FALSE)

# d) 
qchisq(0.995,25, lower.tail = FALSE)



qchisq(0.01,9, lower.tail = FALSE)
qchisq(0.005,9, lower.tail = FALSE)


#*********************       EXERCICIO - BATERIA         #*********************

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

pchisq(0.484,4, lower.tail = FALSE)
pchisq(11.14,4, lower.tail = FALSE)

df <- 4;
chisq <- function(x) (1/((2^(df/2))*gamma(df/2)))*(x^((df/2)-1))*exp(-x/2)
plot(chisq,0,15,lwd=5, col="blue")
polygon(x=c(0.484,seq(0.484,11.143,l=100),11.143), y=c(0,chisq(seq(0.484,11.143,l=100)), 0), col="gray")
abline(h=0,lwd=3,col="black")
abline(v=c(0.484,3.26,11.143), col=c("black","red", "black"), lty=c(1,2,1), lwd=c(1,3,1))
text(c(0.95,3.8, 10.3), c(0.009,0.009, 0.009), c(expression(0.484,3.26,11.143)), col=c("black","red", "black")) 


#*********************       EXERCICIO - LAN?AMENTO         #*********************


pchisq(12,24, lower.tail = FALSE)
pchisq(36,24, lower.tail = FALSE)


qchisq(0.1,15, lower.tail = FALSE)



#..................................................................................


dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

quantil_95 <- qchisq(0.95,10)
quantil_95

df <- 10;
chisq <- function(x) (1/((2^(df/2))*gamma(df/2)))*(x^((df/2)-1))*exp(-x/2)
plot(chisq,0,25,lwd=3, col="black", ylab="Densidade", xlab = "")
mtext("x", side = 1, line = 0.8, at= 26, cex=1.5)
abline(h=0)

x <- seq(quantil_95, 25, 0.001)
lines(x,dchisq(x,10), type="h", col="red")
text(10, 0.04, "?rea de 95%", cex=1.5, font=2)



#..................................................................................

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

# (a) P(??2 150 ??? 126)

1 - pchisq(126,150)

df <- 150
chisq <- function(x) (1/((2^(df/2))*gamma(df/2)))*(x^((df/2)-1))*exp(-x/2)
integrate(chisq, 126, 2000)


# (b) P(40" ??? ??2 65  ??? 50)

pchisq(50, 65) - pchisq(40, 65)

df <- 65
chisq <- function(x) (1/((2^(df/2))*gamma(df/2)))*(x^((df/2)-1))*exp(-x/2)
integrate(chisq, 40, 50)

plot(chisq,0,100,lwd=3, col="black", ylab="Densidade", xlab = "")
mtext("x", side = 1, line = 0.8, at= 104, cex=1.5)
abline(h=0)

x <- seq(40, 55, 0.001)
lines(x,dchisq(x,65), type="h", col="red")
text(35, 0.01, "?rea de ~8%", cex=1.5, font=2)


# (c) P(??2 220 ??? 260)

1 - pchisq(260,220)

df <- 220
chisq <- function(x) (1/((2^(df/2))*gamma(df/2)))*(x^((df/2)-1))*exp(-x/2)
integrate(chisq, 260, 500)



# (d) P(??2 100 ??? a) = 0,6

qchisq(0.6,100)






#####################################################################################
#                                                                                   #
#                                 t - S T U D E N T                                 #
#                                                                                   #
#####################################################################################



#*********************       INTEGRAL         #*********************

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


# FUN??O DENSIDADE DE PROBABILIDADE
df <- 15;
t <- function(x) (gamma((df+1)/2))/((sqrt(df*pi))*(gamma(df/2))*((1+((x^2)/df))^((df+1)/2)))
int <- integrate(t,-Inf,1.753)
alpha <- 1 - int$value
plot(t,-5,5)
int
alpha

polygon(x=c(1.753,seq(1.753,10,l=20),10), y=c(0,t(seq(1.753,10,l=20)), 0), col="gray")

text(c(2.2, 0), c(0.02, 0.2), c(expression(alpha,1-alpha)))


#*********************       CURVA T E COMPARA??O COM A NORMAL         #*********************

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

par(bty="n")

# Plotando a curva T 
x = seq(from=-4, to=4, length=100)
gl = 15     # Graus de Liberdade
alpha = 0.05
curva.t = dt(x, df=gl)

plot(0,0,col="white",xlab="",ylab="",xlim=c(-4,4),ylim=c(0,0.4))
lines(x, curva.t, lty=1, lwd=2)

t.1 <- qt(alpha/2, gl, lower.tail = TRUE)
t.2 <- qt(1-alpha/2, gl, lower.tail = TRUE)
abline(v=c(t.1,t.2), lty=2, lwd=2, col="red")
abline(h=0)

# Curva e ?reas sombreadas
curve(dt(x, gl), from = -4, to = 4,lwd=2, ylab="",ylim=c(-0.05,0.4), labels = FALSE)
coord.x1 <- seq(-4, t.1, len = 100)
coord.y1 <- dt(coord.x1, 15)
polygon(c(coord.x1[1], coord.x1, coord.x1[100]), c(dt(-4, 15), coord.y1, dt(4, 15)),
        col = "grey", border = NA)
coord.x2 <- seq(t.2,4, len = 100)
coord.y2 <- dt(coord.x2, 15)
polygon(c(coord.x2[1], coord.x2, coord.x2[100]), c(dt(-4, 15), coord.y2, dt(4, 15)),
        col = "grey", border = NA)
abline(h=0)


text(c(-2.4,0,2.4), c(0.015,0.2,0.015), c(expression(alpha/2,1-alpha,alpha/2)))
text(c(-2,2.2), c(-0.03,-0.03), c(expression(bar(x)-t[alpha/2]*frac(s,sqrt(n))), 
                                  expression(bar(x)+t[alpha/2]*frac(s,sqrt(n)))))


#________________________________________________________________________________________________________

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

# a)
qt(0.025, 14, lower.tail = FALSE)

# b) 
-qt(0.10, 10, lower.tail = TRUE)

# c)
qt(0.995, 14, lower.tail = TRUE)





pt(-1.761,14, lower.tail = TRUE)
qt(0.005,14, lower.tail = TRUE)


pt(1.697,30, lower.tail = FALSE)
qt(0.05,30, lower.tail = FALSE)


pnorm(1.645, lower.tail = FALSE)
qnorm(0.05, lower.tail = FALSE)


qt(0.05,44, lower.tail = FALSE)


#________________________________________________________________________________________________________



dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

par(mfrow=c(1,3))

par(pty = "s")     # Eixos na mesma escala

x = seq(from=-4, to=4, length=100)
curva.normal = dnorm(x, 0, 1)
gl = 3
curva.t = dt(x, df=gl)

plot(x, curva.normal, type="n", ylab="f(x)", las=1)
lines(x, curva.normal, lty=1, lwd=3,col="blue")
lines(x, curva.t, lty=1, lwd=3,col="red")
legend(0.5, .39, lty=c(1,1), lwd=c(3,3),col=c("blue","red"),box.col = "white",
       legend=c(expression(N(mu == 0,sigma == 1)),paste("t com ", gl,"gl", sep="")))

x = seq(from=-4, to=4, length=100)
curva.normal = dnorm(x, 0, 1)
gl = 5
curva.t = dt(x, df=gl)

plot(x, curva.normal, type="n", ylab="f(x)", las=1)
lines(x, curva.normal, lty=1, lwd=3,col="blue")
lines(x, curva.t, lty=1, lwd=3,col="red")
legend(0.5, .39, lty=c(1,1), lwd=c(3,3),col=c("blue","red"),box.col = "white",
       legend=c(expression(N(mu == 0,sigma == 1)),paste("t com ", gl,"gl", sep="")))

x = seq(from=-4, to=4, length=100)
curva.normal = dnorm(x, 0, 1)
gl = 9
curva.t = dt(x, df=gl)

plot(x, curva.normal, type="n", ylab="f(x)", las=1)
lines(x, curva.normal, lty=1, lwd=3,col="blue")
lines(x, curva.t, lty=1, lwd=3,col="red")
legend(0.5, .39, lty=c(1,1), lwd=c(3,3),col=c("blue","red"),box.col = "white",
       legend=c(expression(N(mu == 0,sigma == 1)),paste("t com ", gl,"gl", sep="")))


#....................................................................................

qt(0.05,44, lower.tail = FALSE)

#....................................................................................

pt(2,4) - pt(-2,4)







#*********************       PT         #*********************

pt()   # Usado para a fun??o de densidade acumulada

pt(1.812461,10, lower.tail = FALSE)

pt(1.812461,10, lower.tail = TRUE)


#*********************       QT         #*********************

qt()   # fornece o quantil q para uma probabilidade p

qt(0.05,10, lower.tail = FALSE)


# a) 
qt(0.025,14, lower.tail = FALSE)

# b) 
-qt(0.10,10, lower.tail = FALSE)


# c) 
qt(0.995,7, lower.tail = FALSE)




#..................................................................................


dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")



# a) Z < 1.25
mi <- 0;
sigma <- 1
phi <- function(x) (1/(sigma*sqrt(2*pi)))*exp((-1/2)*((x-mi)/sigma)^2)
integrate(phi,-Inf,1.25)
plot(phi,-5,5,xlab="", ylab="",lwd=4)




#####################################################################################
#                                                                                   #
#                              F - S N E D E C O R                                 #
#                                                                                   #
#####################################################################################



#*********************       INTEGRAL         #*********************

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


# FUN??O DENSIDADE DE PROBABILIDADE
df1 <- 3;            # graus de liberdade do numerador
df2 <- 30;           # graus de liberdade do denominador  
f <- function(x) (gamma((df1 + df2)/2)/(gamma(df1/2)*gamma(df2/2)))*(df1^(df1/2))*(df2^(df2/2))*(x^((df1/2)-1))*((df1*x+df2)^(-(df1+df2)/2))

int <- integrate(f,0,2.92)
alpha <- 1 - int$value
plot(f,0,5)
int
alpha




#.....................................................................................................

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


# FUN??O DENSIDADE DE PROBABILIDADE
df1 <- 2;            # graus de liberdade do numerador
df2 <- 4;           # graus de liberdade do denominador  
f <- function(x) (gamma((df1 + df2)/2)/(gamma(df1/2)*gamma(df2/2)))*(df1^(df1/2))*(df2^(df2/2))*
                  (x^((df1/2)-1))*((df1*x+df2)^(-(df1+df2)/2))

par(pty = "s")     # Eixos na mesma escala

plot(f, 0, 4,lwd=3)
dfs <- data.frame(c(4,9),c(2,4),c(19,19),c(255,1))
cores <- c("blue", "red", "green","orange")
for (i in 1:4) curve(df(x, df1 = dfs[1,i], df2 = dfs[2,i] ), 0, 4, ylim = c(0,1.2), lwd=3, add = TRUE, 
                     col = cores[i])

#..........................................................................................................

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


# FUN??O DENSIDADE DE PROBABILIDADE
df1 <- 15;            # graus de liberdade do numerador
df2 <- 12;           # graus de liberdade do denominador  
f <- function(x) (gamma((df1 + df2)/2)/(gamma(df1/2)*gamma(df2/2)))*(df1^(df1/2))*
                  (df2^(df2/2))*(x^((df1/2)-1))*((df1*x+df2)^(-(df1+df2)/2))

# par(pty = "s")     # Eixos na mesma escala
plot(f,0,5, lwd=3)

x <- seq(1, 3, 0.001)
lines(x,df(x,df1 = df1, df2 = df2), type="h", col="red")
abline(h=0, col="black", lwd=3, lty=1)


pf(3,15,12) - pf(1,15,12)

#..........................................................................................................

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

t <- qt(0.975, 5)
f <- qf(0.95, 1, 5)

t^2
f


#..........................................................................................................


dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


f___1_menos_alpha___m___n <- qf(0.95, 6, 10, lower.tail = FALSE)
f___1_menos_alpha___m___n


f___alpha___n___m <- qf(0.05, 10, 6, lower.tail = FALSE)
f___alpha___n___m
um_sobre___f___alpha___n___m <- 1/f___alpha___n___m
um_sobre___f___alpha___n___m


#..........................................................................................................



#*********************       EXEMPLO         #*********************

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


# FUN??O DENSIDADE DE PROBABILIDADE
df1 <- 4;            # graus de liberdade do numerador
df2 <- 5;           # graus de liberdade do denominador  
f <- function(x) (gamma((df1 + df2)/2)/(gamma(df1/2)*gamma(df2/2)))*(df1^(df1/2))*(df2^(df2/2))*(x^((df1/2)-1))*((df1*x+df2)^(-(df1+df2)/2))

plot(f,0,6, lwd=3)
abline(h=0, col="black", lwd=3, lty=1)
abline(v=1.4423, col="BLUE", lwd=3, lty=2)

f.critico <- qf(0.05,4,5, lower.tail = FALSE)
f.critico

abline(v=(f.critico), col="red", lwd=3, lty=2)

polygon(x=c(f.critico,seq(f.critico,10,l=20),10), y=c(0,f(seq(f.critico,10,l=20)), 0), col="gray")

text(c(5.8, 0.8), c(0.2, 0.2), c(expression(alpha,1-alpha)))

arrows(5.8, 0.18, 5.4, 0.03,length=0.2,angle=20)

#segments(5.8, 0.18, 5.4, 0.03)


#*********************       AEROPORTO         #*********************

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


# FUN??O DENSIDADE DE PROBABILIDADE
df1 <- 5;            # graus de liberdade do numerador
df2 <- 3;            # graus de liberdade do denominador  
f <- function(x) (gamma((df1 + df2)/2)/(gamma(df1/2)*gamma(df2/2)))*(df1^(df1/2))*
        (df2^(df2/2))*(x^((df1/2)-1))*((df1*x+df2)^(-(df1+df2)/2))

plot(f,0,6, lwd=3)
abline(h=0, col="black", lwd=3, lty=1)
abline(v=2.569866, col="BLUE", lwd=3, lty=2)

f.critico <- qf(0.1,df1,df2, lower.tail = FALSE)
f.critico

abline(v=(f.critico), col="red", lwd=3, lty=2)

polygon(x=c(f.critico,seq(f.critico,10,l=20),10), 
        y=c(0,f(seq(f.critico,10,l=20)), 0), col="gray")

text(c(5.8, 0.8), c(0.2, 0.2), c(expression(alpha,1-alpha)))

arrows(5.8, 0.18, 5.4, 0.03,length=0.2,angle=20)



#*********************       PF         #*********************

pf()   # Usado para a fun??o de densidade acumulada

pf(2.922269,3,30, lower.tail = FALSE)

pf(2.922269,3,30, lower.tail = TRUE)



#*********************       QF         #*********************

qf()   # fornece o quantil q para uma probabilidade p

qf(0.95,5,10, lower.tail = FALSE)


# a) 
qf(0.025,5,10, lower.tail = FALSE)

# b) 
qt(0.10,24,9, lower.tail = FALSE)

# c) 
qt(0.95,8,15, lower.tail = FALSE)




#####################################################################################
#                                                                                   #
#                          TRIGLICERIDEOS MULHERES                                  #
#                                                                                   #
#####################################################################################


###################################   T G R  ########################################   

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


options(max.print=100)    # Mostra no máximo 100 valores


mulheres <- read.table("D:/MAURI/2023/CURSOS/POUPEX___2023/AULA___4/DADOS___SANGUINEOS___MULHERES.txt",
                       sep = "" , header = TRUE)



mulheres[, 1]
tgr <- mulheres[, 1]
max(tgr)
min(tgr)
length(tgr)             # TAMANHO DO VETOR TGR
hist(tgr,main="TGR", prob=TRUE)

lines(density(tgr))
dens.tgr <- density(tgr)


library(fitdistrplus)
library(logspline)

descdist(tgr, discrete = FALSE)

tgr_normalizado <- (tgr - min(tgr))/(max(tgr) - min(tgr))
tgr_normalizado

fit.weibull <- fitdist(tgr, "weibull")
fit.gamma <- fitdist(tgr, "gamma")
fit.lognormal <- fitdist(tgr, "lnorm")
fit.normal <- fitdist(tgr, "norm")
fit.beta <- fitdist(tgr_normalizado, "beta", method = "mme")

summary(fit.beta)
fit.beta$aic

plot(fit.weibull)
plot(fit.gamma)
plot(fit.lognormal)
plot(fit.normal)

fit.weibull$aic
fit.gamma$aic            # Crit?rio de Informa??o de Akaike (AIC)
fit.lognormal$aic
fit.normal$aic


mle_tgr <- fitdist(tgr,"lnorm",method="mle")
summary(mle_tgr)

ks.test(tgr, "plnorm", meanlog=4.6116562, sdlog=0.5439875, exact=FALSE)


###################################   A P B  ######################################## 


dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


options(max.print=100)    # Mostra no máximo 100 valores

mulheres <- read.table("D:/MAURI/2023/CURSOS/POUPEX___2023/AULA___4/DADOS___SANGUINEOS___MULHERES.txt",
                       sep = "" , header = TRUE)

mulheres[, 3]
apb <- mulheres[, 3]
max(apb)
min(apb)
length(apb)             # TAMANHO DO VETOR TGR
hist(apb,main="TGR", prob=TRUE)

lines(density(apb))
dens.tgr <- density(apb)


library(fitdistrplus)
library(logspline)

descdist(apb, discrete = FALSE)

fit.weibull <- fitdist(apb, "weibull")
fit.gamma <- fitdist(apb, "gamma")
fit.lognormal <- fitdist(apb, "lnorm")
fit.normal <- fitdist(apb, "norm")


plot(fit.weibull)
plot(fit.gamma)
plot(fit.lognormal)
plot(fit.normal)

fit.weibull$aic
fit.gamma$aic            # Critério de Informação de Akaike
fit.lognormal$aic
fit.normal$aic


mle_apb <- fitdist(apb,"gamma",method="mle")
summary(mle_apb)

# Kolmogorov-Smirnov
ks.test(apb, "pgamma", 11.573742, 0.123561, exact=FALSE)

table(apb)




library(goft)        
gamma_test(apb)

library(EnvStats)
distChoose(apb)



#####################################################################################
#                                                                                   #
#  E X E R C Í C I O     I D E N T I F I C A C A O       D I S T R I B U I C O E S  #
#                                                                                   #
#####################################################################################


dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")


#gama_sim <- rgamma(30,shape = 3, scale = 2)
#gama_sim

library(fitdistrplus)
library(logspline)

dados_gamma <- c(20.8625807,  7.2445709,  4.4659396,  3.2712081,  4.9300651,  5.7444213,  6.6700987,
                 11.1750446,  2.3753017,  3.5425386,  0.5978486,  6.8869953,  6.1102197,  8.2716973,
                 9.7465462,  3.3991988,  1.8557047, 11.3983705,  3.6847590,  2.3327479,  6.1364329,
                 4.4686122,  7.8007834,  4.7649257,  3.8829371,  5.9986131,  5.5163819,  9.6951710,
                 10.1645820,  6.1304865)

# Identificação da Distribuição
hist(dados_gamma, breaks = 20)

descdist(dados_gamma, discrete = FALSE)


fit.weibull <- fitdist(dados_gamma, "weibull")
fit.gamma <- fitdist(dados_gamma, "gamma")
fit.lognormal <- fitdist(dados_gamma, "lnorm")
fit.normal <- fitdist(dados_gamma, "norm")

plot(fit.weibull)
plot(fit.gamma)
plot(fit.lognormal)
plot(fit.normal)

fit.weibull$aic
fit.gamma$aic            # Critério de Informação de Akaike
fit.lognormal$aic
fit.normal$aic


mle_dados_gamma <- fitdist(dados_gamma, "gamma", method = "mle")
summary(mle_dados_gamma)

# Kolmogorov-Smirnov
#ks.test(mle_dados_gamma, "pgamma", shape = 2.8718513  , rate = 0.4555631, exact=FALSE)
ks.test(dados_gamma, "pgamma", 2.8718513  , 0.4555631, exact = FALSE)

# Construir a Função
alpha <- mle_dados_gamma$estimate[1]; #alpha <- 2.8718513
lambda <- mle_dados_gamma$estimate[2]; #lambda <- 0.4555631
  
f_gamma <- function(x) {((lambda^alpha)/(gamma(alpha)))*(x^(alpha - 1))*(exp(-lambda*x))}

# Plotar a Função e o Histograma
max(dados_gamma)
min(dados_gamma)

curve(f_gamma, from = 0, to = 22, xlab = "x", ylab = "y")
integrate(f_gamma, 0, Inf)

hist(dados_gamma, ylim = c(0, 0.125), freq = FALSE)
curve(f_gamma, from = 0, to = 22, xlab = "x", ylab = "y", add = TRUE)

curve(f_gamma, from = 0, to = 22, xlab = "x", ylab = "y")
polygon(x=c(0, seq(0, 10, l=100), 10), y=c(0,f_gamma(seq(0, 10, l=100)), 0), col="blue")

# Cáculo de áreas sob a função
# 1
curve(f_gamma, from = 0, to = 22, xlab = "x", ylab = "y")
polygon(x=c(0, seq(0, 10, l=100), 10), y=c(0,f_gamma(seq(0, 10, l=100)), 0), col="darkolivegreen2")
integrate(f_gamma, 0, 10)

pgamma(10, 2.8718513, 0.455563)

# 2
curve(f_gamma, from = 0, to = 22, xlab = "x", ylab = "y")
polygon(x=c(10, seq(10, 15, l=100), 15), y=c(0,f_gamma(seq(10, 15, l=100)), 0), col="indianred1")
integrate(f_gamma, 10, 15)
#abline(h = 0)

pgamma(15, 2.8718513, 0.455563) - pgamma(10, 2.8718513, 0.455563)


pnorm(1.25, 0, 1)
qnorm(0.8943502, 0, 1)
dnorm(1.25, 0, 1)
rnorm(30, 0, 1)



#______________________
dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

library(fitdistrplus)
library(logspline)

#weibull_sim <- rweibull(30,shape = 2, scale = 2)
#weibull_sim                                       # Comandos usados para inicialmente
#descdist(weibull_sim, discrete = FALSE)           # simular uma Weibull


dados_weibull <- c(1.4940354, 2.0164275, 1.9513521, 1.5298282, 0.6815670, 2.4267801, 0.6762800, 1.7018986,
4.1632638, 2.5472784, 2.2174151, 0.6058986, 1.7432601, 1.1199216, 1.7135932, 2.8758657,
0.8537880, 1.5511504, 2.3262178, 2.3267933, 1.3916375, 4.7439947, 2.1864812, 2.0269031,
1.7489244, 1.8191036, 2.0845146, 1.2229195, 1.0115042, 2.7931222)

# Identificação da Distribuição
hist(dados_weibull, freq = FALSE)

descdist(dados_weibull, discrete = FALSE)


fit.weibull <- fitdist(dados_weibull, "weibull")
fit.gamma <- fitdist(dados_weibull, "gamma")
fit.lognormal <- fitdist(dados_weibull, "lnorm")
fit.normal <- fitdist(dados_weibull, "norm")

plot(fit.weibull)
plot(fit.gamma)
plot(fit.lognormal)
plot(fit.normal)

fit.weibull$aic
fit.gamma$aic            # Critério de Informação de Akaike
fit.lognormal$aic
fit.normal$aic


mle_dados_weibull <- fitdist(dados_weibull,"weibull", method="mle")
summary(mle_dados_weibull)

mle_weibull_ou_gamma <- fitdist(dados_weibull,"gamma", method="mle")
summary(mle_weibull_ou_gamma)

mle_weibull_ou_lnorm <- fitdist(dados_weibull,"lnorm", method="mle")
summary(mle_weibull_ou_lnorm)

# Kolmogorov-Smirnov
ks.test(dados_weibull, "pweibull", 2.231941 , 2.169857, exact=FALSE)

ks.test(dados_weibull, "pgamma", 4.697015, 2.448444, exact=FALSE)

ks.test(dados_weibull, "plnorm", 0.5412805, 0.4821805, exact=FALSE)


# Construi uma simulação para obter dados de uma distribuição Weibull, mas pelos dados 
# gerados o AIC da Gamma é menor e não rejeito que seja uma Gamma.
# Os comandos a seguir são usados para tentar gerar uma Weibull que tenha menor AIC e 
# que o Teste de Kolmogorov-Smirnov não rejeite ser Weibull

dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

library(fitdistrplus)
library(logspline)

#weibull_sim <- rweibull(30,shape = 7, scale = 2)
#weibull_sim                                       # Comandos usados para inicialmente
#descdist(weibull_sim, discrete = FALSE)           # simular uma Weibull


dados_weibull <- c(1.9993382, 1.4414849, 2.1477166, 2.1087828, 2.1342892, 2.1844835, 1.5091879, 2.0467623, 1.0642741,
                   2.1302612, 1.8389897, 1.8924614, 1.9316041, 1.5602204, 1.6991884, 1.7228081, 1.5197833, 1.7659242,
                   0.6914335, 1.4598759, 2.0017607, 1.5139209, 1.8334780, 1.8847480, 1.9072389, 1.6294414, 1.9068617,
                   1.7744973, 2.4300455, 1.8958270)
  


# Identificação da Distribuição
hist(dados_weibull, freq = FALSE)

descdist(dados_weibull, discrete = FALSE)


fit.weibull <- fitdist(dados_weibull, "weibull")
fit.gamma <- fitdist(dados_weibull, "gamma")
fit.lognormal <- fitdist(dados_weibull, "lnorm")
fit.normal <- fitdist(dados_weibull, "norm")
fit.uniform <- fitdist(dados_weibull, "unif")

plot(fit.weibull)
plot(fit.gamma)
plot(fit.lognormal)
plot(fit.normal)

fit.weibull$aic
fit.gamma$aic            # Critério de Informação de Akaike
fit.lognormal$aic
fit.normal$aic
fit.uniform$aic


mle_dados_weibull <- fitdist(dados_weibull,"weibull", method="mle")
summary(mle_dados_weibull)

mle_weibull_ou_gamma <- fitdist(dados_weibull,"gamma", method="mle")
summary(mle_weibull_ou_gamma)

mle_weibull_ou_lnorm <- fitdist(dados_weibull,"lnorm", method="mle")
summary(mle_weibull_ou_lnorm)

# Kolmogorov-Smirnov
ks.test(dados_weibull, "pweibull", 6.513198, 1.918411, exact=FALSE)

ks.test(dados_weibull, "pgamma", 20.98456, 11.73920, exact=FALSE)

ks.test(dados_weibull, "plnorm", 0.5568348, 0.2367576, exact=FALSE)

# Construir a Função
alpha <- mle_dados_weibull$estimate[1]
beta <- mle_dados_weibull$estimate[2]

f_weibull <- function(x) {(alpha/(beta^alpha))*(x^(alpha-1))*(exp(-(x/beta)^alpha))}

# Plotar a Função e o Histograma
max(dados_weibull)
min(dados_weibull)

curve(f_weibull, from = 0, to = 5, xlab = "x", ylab = "y")
integrate(f_weibull, 0, Inf)

hist(dados_weibull, ylim = c(0, 1.5), freq = FALSE)
curve(f_weibull, from = 0, to = 3, xlab = "x", ylab = "y", add = TRUE)

curve(f_weibull, from = 0, to = 3, xlab = "x", ylab = "y")
polygon(x=c(1, seq(1, 1.5, l=100), 1.5), y=c(0,f_weibull(seq(1, 1.5, l=100)), 0), col="blue")

# Cálculo de áreas sob a função
# 1
curve(f_weibull, from = 0, to = 3, xlab = "x", ylab = "y")
polygon(x=c(1, seq(1, 1.5, l=100), 1.5), y=c(0,f_weibull(seq(1, 1.5, l=100)), 0), col="darkolivegreen2")
integrate(f_weibull, 1, 1.5)

pweibull(1.5, 6.513198, 1.918411) - pweibull(1, 6.513198, 1.918411)


library(goft)        
weibull_test(dados_weibull)

library(EnvStats)
distChoose(dados_weibull)

eweibull(dados_weibull, method = "mle")   # O eweibull também está no EnvStats


#______________________
dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

library(fitdistrplus)
library(logspline)

#logn_sim <- rlnorm(30, meanlog = 3, sdlog = 1)
#logn_sim
#descdist(logn_sim, discrete = FALSE)


dados_lnorm <- c(9.534149, 12.878719, 35.635908, 39.158389, 10.091099, 133.714299, 15.684000, 3.179206,
                   16.073085, 57.767201, 29.543033, 24.672685, 11.955565, 2.132028, 17.455254, 20.569096,
                   6.293823, 22.717485, 83.353863, 18.544482, 66.437399, 4.616951, 18.931367, 1.464430,
                   21.180916, 179.315876, 24.941790, 14.105447, 7.680880,17.688369)



# Identificação da Distribuição
hist(dados_lnorm, freq = FALSE)

descdist(dados_lnorm, discrete = FALSE)


fit.weibull <- fitdist(dados_lnorm, "weibull")
fit.gamma <- fitdist(dados_lnorm, "gamma")
fit.lognormal <- fitdist(dados_lnorm, "lnorm")
fit.normal <- fitdist(dados_lnorm, "norm")
fit.uniform <- fitdist(dados_lnorm, "unif")

plot(fit.weibull)
plot(fit.gamma)
plot(fit.lognormal)
plot(fit.normal)

fit.weibull$aic
fit.gamma$aic            # Critério de Informação de Akaike
fit.lognormal$aic
fit.normal$aic
fit.uniform$aic


mle_dados_lnorm <- fitdist(dados_lnorm,"lnorm", method="mle")
summary(mle_dados_lnorm)

# Kolmogorov-Smirnov
ks.test(dados_lnorm, "plnorm", 2.869537, 1.079718, exact=FALSE)

# Construir a Função
meanlog <- mle_dados_lnorm$estimate[1]
sdlog  <- mle_dados_lnorm$estimate[2]

f_lnorm <- function(x) {(1/(((2*pi)^0.5)*(sdlog*x)))*exp(-((log(x) - meanlog)^2)/(2*(sdlog^2)))}

# Plotar a Função e o Histograma
max(dados_lnorm)
min(dados_lnorm)

curve(f_lnorm, from = 0, to = 200, xlab = "x", ylab = "y")
integrate(f_lnorm, 0, Inf)

hist(dados_lnorm, ylim = c(0, 0.036), breaks = 20, freq = FALSE)
curve(f_lnorm, from = 0, to = 200, xlab = "x", ylab = "y", add = TRUE)

curve(f_lnorm, from = 0, to = 200, xlab = "x", ylab = "y")
polygon(x=c(40, seq(40, 50, l=100), 50), y=c(0,f_lnorm(seq(40, 50, l=100)), 0), col="blue")

# Cáculo de áreas sob a função
# 1
curve(f_lnorm, from = 0, to = 200, xlab = "x", ylab = "y")
polygon(x=c(40, seq(40, 50, l=100), 50), y=c(0,f_lnorm(seq(40, 50, l=100)), 0), col="darkolivegreen2")
integrate(f_lnorm, 40, 50)

plnorm(50, 2.869537, 1.079718) - plnorm(40, 2.869537, 1.079718)



#______________________
dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

library(fitdistrplus)
library(logspline)

#norm_sim <- rnorm(30, 8, 2)
#norm_sim


dados_norm <- c(4.391658,  5.364267, 10.707930,  5.431008,  6.904122,  6.960462, 12.741468,  8.094473,  7.255829,
8.434530,  9.747057,  6.440681,  7.623020,  9.276933,  8.711818,  5.250229,  6.482474,  3.478216,
9.717008,  9.317296,  9.011653, 11.758927, 10.844472,  9.644711,  7.541715,  7.561009, 10.034726,
9.654606,  6.222452,  5.207637)

descdist(dados_norm, discrete = FALSE)

fit.weibull <- fitdist(dados_norm, "weibull")
fit.gamma <- fitdist(dados_norm, "gamma")
fit.lognormal <- fitdist(dados_norm, "lnorm")
fit.normal <- fitdist(dados_norm, "norm")
fit.uniform <- fitdist(dados_norm, "unif")

#plot(fit.weibull)
#plot(fit.gamma)
#plot(fit.lognormal)
#plot(fit.normal)

fit.weibull$aic
fit.gamma$aic            # Critério de Informação de Akaike
fit.lognormal$aic
fit.normal$aic
fit.uniform$aic


mle_dados_weibull <- fitdist(dados_norm,"weibull", method="mle")
summary(mle_dados_weibull)

mle_dados_norm <- fitdist(dados_norm,"norm", method="mle")
summary(mle_dados_norm)

# Kolmogorov-Smirnov
ks.test(dados_norm, "pweibull", 4.057284 , 8.819386, exact=FALSE)

ks.test(dados_norm, "pnorm", 7.993746 , 2.208748, exact=FALSE)


# Construir a Função
mi <- mle_dados_norm$estimate[1]
sigma  <- mle_dados_norm$estimate[2]

# 1
phi1 <- function(x) (1/(sigma*sqrt(2*pi)))*exp((-1/2)*((x-mi)/sigma)^2)
integrate(phi1,-Inf,10)
phi1(10)
plot(phi1,0,16)

# 2
phi2 <- function(x, mi, sigma) (1/(sigma*sqrt(2*pi)))*exp((-1/2)*((x-mi)/sigma)^2)
phi2(10, 7.993746, 2.208748)
integrate(phi2,-Inf, 10, mi = 7.993746, sigma = 2.208748)

# Plotar a Função e o Histograma
max(dados_norm)
min(dados_norm)

curve(phi1, from = 0, to = 16, xlab = "x", ylab = "y")
integrate(phi1, -Inf, 10)

hist(dados_norm, ylim = c(0, 0.20), freq = FALSE)
curve(phi1, from = 0, to = 16, xlab = "x", ylab = "y", add = TRUE)

curve(phi1, from = 0, to = 16, xlab = "x", ylab = "y")
polygon(x=c(0, seq(0, 10, l=100), 10), y=c(0, phi1(seq(0, 10, l=100)), 0), col="blue")

# Cáculo de áreas sob a função
curve(phi1, from = 0, to = 16, xlab = "x", ylab = "y")
polygon(x=c(0, seq(0, 10, l=100), 10), y=c(0, phi1(seq(0, 10, l=100)), 0), col="moccasin")
integrate(phi1, -Inf, 10)

pnorm(10, 7.993746, 2.208748) - pnorm(-Inf, 7.993746, 2.208748)




#______________________
dev.off(dev.list()["RStudioGD"])
rm(list=ls())
cat("\f")

library(fitdistrplus)
library(logspline)

# norm_sim <- rnorm(8, 4, 1)
# norm_sim

dados_norm <- c(3.816942, 4.123619, 4.575150, 3.214129, 4.854917, 3.647232, 4.003734, 3.261923)

descdist(dados_norm, discrete = FALSE)










