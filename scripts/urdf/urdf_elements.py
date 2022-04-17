# This is a script to generate URDF models for ROS simulation in rviz
# Maintainer: Julian Gaal github.com/juliangaal
import textwrap

class File:
	"""Create XML file for ROS urdf model.
    Args:
        'filename': name of file to be saved to. Should end with .urdf.xacro.
        'objects': individual elements, e.g. link, xacro property, joint
    Raises:
       	No Error yet
    """
	def __init__(self, *objects, filename='example.urdf.xacro'):
		self.objects = objects
		self.xml = []
		self.filename = filename

	def __complete_xml(self):
		self.xml.insert(0, '<?xml version="1.0" ?>')
		self.xml.insert(0, '<robot>')
		self.xml.insert(len(self.xml), '</robot>')
		return self.xml

	def save(self):
		with open(self.filename, "w") as f:
			for key, val in enumerate(self.objects):
				f.write(val.element)

		print("\t-- Success -- \nSaved to %s" % self.filename)
		f.close()

class Property:
	def __init__(self, name, value):
		self.name = name
		self.value = value
		self.element = '<xacro:property name="%s" value="%s"/>\n' % (self.name, self.value)

class Link:
	"""Create 'link' urdf element.
    Args:
        rgba: red - green - blue - alpha.
        size: size of object.
    Raises:
       	No Error yet
    """
	def __init__(self, typ, name, rgba, size, color, rpy='0 0 0'):
		self.rpy = rpy
		self.type = typ
		self.name = name
		self.rgba = rgba
		self.size = size
		self.color = color
		self.element = self.xmlify()
		
	# Formatting options, see http://tinyurl.com/y9vohsxz

	# Manual xml string, as backup: '#'\n\n<link name="test1">\n\t<visual>\n\t\t<geometry>\n\t\t\t<box size="0.6 0.3 0.2"/>\n\t\t</geometry>
	# \n\t\t<material name="green">\n\t\t\t<color rgba="0.1 0.3 0.2"/>\n\t\t</material>\n\t</visual>\n</link>''

	# ISSUE: String after closing brackets of material need to be indented one space more. wtf
	def xmlify(self):
		xml = '\n'.join([
				'\n<link name="%s">' % self.name,
				'  <visual>',
				'    <geometry>',
				'      <%s size="%s" rpy="%s"/>' 	% (self.type, self.size, self.rpy),
				'    </geometry>',
				'	<material name="%s">' 	% self.color,
				'	  <color rgba="%s"/>' % self.rgba,
				'	</material>',
				'  </visual>',
				'</link>\n'
			])

		return xml

class Joint:
	def __init__(self, name, typ, parent, child, xyz, rpy='0 0 0'):
		self.name = name
		self.type = typ
		self.parent = parent
		self.child = child
		self.xyz = xyz
		self.rpy = rpy
		self.element = self.xmlify()

	def xmlify(self):
		xml = '\n'.join([
			'\n<joint name="%s">' % self.name,
			'  <parent link="%s"/>' % self.parent,
			'  <child link="%s"/>' % self.child,
			'  <origin xyz="%s" rpy="%s"/>' % (self.xyz, self.rpy),
			'</joint>\n'
			])

		return xml