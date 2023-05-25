

from ..graphs import create_graph, Phenomenon, node, hparam, register_graph


@register_graph('example')
class Example(Phenomenon):
	description_template = hparam('{Zname} mediates the effect of {Xname} on {Yname}.')

	X = node()
	Z = node(X)
	Y = node(X, Z)



@register_graph('example-confounding')
class ExampleCon(Phenomenon):
	description_template = hparam('{Zname} confounds the effect of {Xname} on {Yname}.')

	Z = node()
	X = node(Z)
	Y = node(X, Z)



@register_graph('example2')
class Example2(Phenomenon):
	description_template = hparam('{Xname} causes {Yname}.')

	X = node()
	Y = node(X)



@register_graph('example3')
class Example3(Phenomenon):
	description_template = hparam('{Xname} and mutually indpendent {Zname} but both cause {Yname}.')

	X = node()
	Z = node()
	Y = node(X, Z)



example_labels = {
	'Xname': 'X',
	'Zname': 'Z',
	'Yname': 'Y',
	'Wname': 'W',
	'V1name': 'V1',
	'V2name': 'V2',
	'V3name': 'V3',

	'X0': 'X=0',
	'X1': 'X=1',

	'Z0': 'Z=0',
	'Z1': 'Z=1',

	'Y0': 'Y=0',
	'Y1': 'Y=1',

	'W0': 'W=0',
	'W1': 'W=1',

	'V10': 'V1=0',
	'V11': 'V1=1',

	'V20': 'V2=0',
	'V21': 'V2=1',

	'V30': 'V3=0',
	'V31': 'V3=1',


	'X0_noun': 'X=0',
	'X1_noun': 'X=1',

	'Z0_noun': 'Z=0',
	'Z1_noun': 'Z=1',

	'Y0_noun': 'Y=0',
	'Y1_noun': 'Y=1',

	'W0_noun': 'W=0',
	'W1_noun': 'W=1',

	'V10_noun': 'V1=0',
	'V11_noun': 'V1=1',

	'V20_noun': 'V2=0',
	'V21_noun': 'V2=1',

	'V30_noun': 'V3=0',
	'V31_noun': 'V3=1',


	'X0_sentence': 'X=0',
	'X1_sentence': 'X=1',

	'Z0_sentence': 'Z=0',
	'Z1_sentence': 'Z=1',

	'Y0_sentence': 'Y=0',
	'Y1_sentence': 'Y=1',

	'W0_sentence': 'W=0',
	'W1_sentence': 'W=1',

	'V10_sentence': 'V1=0',
	'V11_sentence': 'V1=1',

	'V20_sentence': 'V2=0',
	'V21_sentence': 'V2=1',

	'V30_sentence': 'V3=0',
	'V31_sentence': 'V3=1',


	'X0_sentence_condition': 'X=0',
	'X1_sentence_condition': 'X=1',

	'Z0_sentence_condition': 'Z=0',
	'Z1_sentence_condition': 'Z=1',

	'Y0_sentence_condition': 'Y=0',
	'Y1_sentence_condition': 'Y=1',

	'W0_sentence_condition': 'W=0',
	'W1_sentence_condition': 'W=1',

	'V10_sentence_condition': 'V1=0',
	'V11_sentence_condition': 'V1=1',

	'V20_sentence_condition': 'V2=0',
	'V21_sentence_condition': 'V2=1',

	'V30_sentence_condition': 'V3=0',
	'V31_sentence_condition': 'V3=1',


	'X0_wheresentence': 'X=0',
	'X1_wheresentence': 'X=1',

	'Z0_wheresentence': 'Z=0',
	'Z1_wheresentence': 'Z=1',

	'Y0_wheresentence': 'Y=0',
	'Y1_wheresentence': 'Y=1',

	'W0_wheresentence': 'W=0',
	'W1_wheresentence': 'W=1',

	'V10_wheresentence': 'V1=0',
	'V11_wheresentence': 'V1=1',

	'V20_wheresentence': 'V2=0',
	'V21_wheresentence': 'V2=1',

	'V30_wheresentence': 'V3=0',
	'V31_wheresentence': 'V3=1',


	'X0_wherepartial': 'X=0',
	'X1_wherepartial': 'X=1',

	'Z0_wherepartial': 'Z=0',
	'Z1_wherepartial': 'Z=1',

	'Y0_wherepartial': 'Y=0',
	'Y1_wherepartial': 'Y=1',

	'W0_wherepartial': 'W=0',
	'W1_wherepartial': 'W=1',

	'V10_wherepartial': 'V1=0',
	'V11_wherepartial': 'V1=1',

	'V20_wherepartial': 'V2=0',
	'V21_wherepartial': 'V2=1',

	'V30_wherepartial': 'V3=0',
	'V31_wherepartial': 'V3=1',

}

