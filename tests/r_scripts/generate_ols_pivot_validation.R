set.seed(2024)
rustvec <- function(name, v) cat(sprintf("const %s: [f64; %d] = [%s];\n", name, length(v),
  paste(ifelse(is.na(v),"f64::NAN",sprintf("%.12g", v)), collapse=", ")))
emit <- function(tag, x, y, intercept) {
  p<-ncol(x); n<-nrow(x); df<-as.data.frame(x); names(df)<-paste0("v",1:p); df$y<-y
  form<-if(intercept) as.formula(paste("y ~",paste0("v",1:p,collapse="+"))) else as.formula(paste("y ~ 0 +",paste0("v",1:p,collapse="+")))
  m<-lm(form,df); s<-summary(m); alln<-c(if(intercept)"(Intercept)",paste0("v",1:p))
  co<-coef(m)[alln]; sev<-rep(NA,length(alln)); sm<-s$coefficients
  for(k in seq_along(alln)){idx<-which(rownames(sm)==alln[k]); if(length(idx))sev[k]<-sm[idx,2]}
  cat(sprintf("// %s: pivot order = %s (1=first col)\n", tag, paste(m$qr$pivot, collapse=",")))
  rustvec(paste0(tag,"_X"), as.numeric(t(x))); rustvec(paste0(tag,"_Y"), y)
  rustvec(paste0(tag,"_COEF"), as.numeric(co)); rustvec(paste0(tag,"_SE"), as.numeric(sev))
  rustvec(paste0(tag,"_FITTED"), as.numeric(fitted(m)))
  cat(sprintf("const %s_N: usize=%d; const %s_P: usize=%d; const %s_SIGMA: f64=%.12g;\n\n",tag,n,tag,p,tag,s$sigma))
}
# Moderate, well-conditioned but non-identity pivot orders.
n<-50; sc<-c(1,12,3,40,7); xA<-sapply(sc,function(s)runif(n,-1,1)*s+s); 
yA<-4+xA%*%c(2.5,-1.2,0.8,3.1,-0.4)+rnorm(n,0,1.0); emit("A",xA,as.numeric(yA),TRUE)
n<-45; sc<-c(50,2,15,1,30,8); xB<-sapply(sc,function(s)runif(n,-1,1)*s+s)
yB<-xB%*%c(1.1,-0.7,2.2,0.5,-1.9,0.3)+rnorm(n,0,0.8); emit("B",xB,as.numeric(yB),FALSE)
# rank-deficient, moderate scale
n<-60; v1<-runif(n,-1,1)*10+20; v2<-runif(n,-1,1)*3+5; v3<-2*v1; v4<-runif(n,-1,1)*8+10
xC<-cbind(v1,v2,v3,v4); yC<-3+xC%*%c(1.5,-2.0,0,0.9)+rnorm(n,0,0.5); emit("C",xC,as.numeric(yC),TRUE)
