from modelo import Model as DenseModel

def Model(kind,arm,lut,checkpointing=True):
 if arm!='residual':return DenseModel(kind,arm,lut,checkpointing)
 from residual_consulta import ResidualModel
 return ResidualModel(kind,lut)
